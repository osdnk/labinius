//! The rokoko side of the opening: ring elements, the outer commitments `T_Y` and `T_u` under
//! tensor keys, the claims of `docs/rokoko.md`, and the proof through rokoko's claims sumcheck
//! and commitment chain.
//!
//! Transcript: `HashWrapper::new()`, the round digest, `T_Y`, `T_u`, the rokoko commitment; then
//! the claims, both sides drawing at the same states: `rho` and one claim per chain; one per key
//! row (`T_Y` rows, then `T_u`); three support claims, each drawing `N` scalars; one norm claim
//! per vector; one binariness claim per binary vector. `prove_claims` absorbs the claim values.
//!
//! Key row `i` of `T_Y` is one uniform tensor point per residue region, so `T_Y` is one
//! structured Ajtai commitment over the union of the regions; `T_u` the same over the lift.
//!
//! Per round the tables cost one lane-wise multiply-add per entry: [`Fixed`] holds the NTT form
//! of the key's polynomials, the round transforms only its own, and `rho^a * poly` is shared
//! by every entry of the round that reads it.
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;
use std::time::Instant;

use rokoko::common::config::MOD_Q;
use rokoko::common::hash::HashWrapper;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::common::sampling::AesCtrPublicSampler;
use rokoko::common::structured_row::PreprocessedRow;
use rokoko::hexl::bindings::eltwise_fma_mod;
use rokoko::protocol::config::{SizeableProof, SumcheckConfig, SumcheckRoundProof};
use rokoko::protocol::crs::{VerifierCRS, CRS};
use rokoko::protocol::parties::{commiter::commit, prover::prover_round, verifier::verifier_round};
use rokoko::protocol::snark::{
    eq, prove_claims, table, verify_claims, witness, witness_in, Claim, ClaimExpr, ClaimsProof,
    Region, WitnessShape,
};
use rokoko::protocol::sumcheck::init_sumcheck;
use rokoko::protocol::sumchecks::builder_verifier::init_verifier;

use super::{Cap, Element, Layout, Poly, Relation, Witness, DEG, SUPPORT};

fn stamp(label: &str, clock: Instant) {
    if std::env::var_os("ROKOKO_TIMINGS").is_some() {
        eprintln!(
            "rokoko-timings {label} {:.1} ms",
            clock.elapsed().as_secs_f64() * 1e3
        );
    }
}

pub const SUPPORT_CLAIMS: usize = 3;

fn zero() -> RingElement {
    RingElement::zero(Representation::IncompleteNTT)
}

fn one() -> RingElement {
    RingElement::one(Representation::IncompleteNTT)
}

fn centred(v: u64) -> i64 {
    if v > MOD_Q / 2 {
        v as i64 - MOD_Q as i64
    } else {
        v as i64
    }
}

fn placed(coefficients: &[i64]) -> RingElement {
    assert!(
        coefficients.len() <= DEG,
        "polynomial of degree {} in a ring of degree {DEG}",
        coefficients.len()
    );
    let mut r = RingElement::new(Representation::Coefficients);
    for (slot, &c) in r.v.iter_mut().zip(coefficients) {
        *slot = c.rem_euclid(MOD_Q as i64) as u64;
    }
    r.to_representation(Representation::IncompleteNTT);
    r
}

/// A committed element as a ring element: coefficients centred mod `q`, `IncompleteNTT`.
pub fn ring(e: &Element) -> RingElement {
    placed(e)
}

/// The inverse of [`ring`]: coefficients centred in `(-q/2, q/2]`.
pub fn coefficients(r: &RingElement) -> Element {
    let mut c = r.clone();
    c.to_representation(Representation::Coefficients);
    core::array::from_fn(|i| centred(c.v[i]))
}

/// A public weight polynomial, low coefficient first, as a ring element.
pub fn weight(p: &Poly) -> RingElement {
    placed(p)
}

/// The constant coefficient of an `IncompleteNTT` element, centred.
pub fn constant_term(r: &RingElement) -> i64 {
    centred(r.constant_term_from_incomplete_ntt())
}

/// Every element the transcript absorbs or a check reads is `IncompleteNTT`: the tag itself is
/// never hashed.
fn ntt<'a>(elements: impl IntoIterator<Item = &'a RingElement>) -> Result<(), String> {
    match elements
        .into_iter()
        .position(|e| e.representation != Representation::IncompleteNTT)
    {
        Some(i) => Err(format!("shipped element {i} is not in NTT representation")),
        None => Ok(()),
    }
}

fn owners(layout: &Layout) -> Vec<Option<usize>> {
    let mut owner = vec![None; layout.len];
    for (v, (vector, region)) in layout.vectors.iter().zip(&layout.regions).enumerate() {
        for slot in &mut owner[region.start..region.start + vector.used] {
            *slot = Some(v);
        }
    }
    owner
}

fn shape(layout: &Layout, config: &SumcheckConfig) -> Result<(usize, usize), String> {
    let n = config.witness_height * config.witness_width;
    if n < layout.len {
        return Err(format!(
            "the chain holds {n} elements, the layout needs {}",
            layout.len
        ));
    }
    if config.nof_openings != 2 {
        return Err(format!(
            "the chain is compiled for {} openings, the claims conjugate",
            config.nof_openings
        ));
    }
    for (v, (vector, region)) in layout.vectors.iter().zip(&layout.regions).enumerate() {
        let aligned = region.len.is_power_of_two() && region.start % region.len == 0;
        if !aligned || region.start + region.len > layout.len || vector.used > region.len {
            return Err(format!(
                "vector {v} ({}) is not a region of the layout",
                vector.name
            ));
        }
    }
    Ok((config.witness_height, config.witness_width))
}

fn region(layout: &Layout, v: usize, total: usize) -> Region {
    let r = layout.regions[v];
    Region::new(r.start, r.len, total)
}

/// Every vector at its region inside the chain's `N` elements, zero elsewhere.
pub fn committed(
    layout: &Layout,
    witness: &Witness,
    config: &SumcheckConfig,
) -> Result<VerticallyAlignedMatrix<RingElement>, String> {
    let (height, width) = shape(layout, config)?;
    if witness.len() != layout.vectors.len() {
        return Err(format!(
            "{} vectors for {} regions",
            witness.len(),
            layout.vectors.len()
        ));
    }
    let mut data = vec![zero(); height * width];
    for (v, (vector, region)) in layout.vectors.iter().zip(&layout.regions).enumerate() {
        if witness[v].len() != vector.used {
            return Err(format!(
                "vector {v} ({}) has {} elements, the layout says {}",
                vector.name,
                witness[v].len(),
                vector.used
            ));
        }
        for (slot, e) in data[region.start..].iter_mut().zip(&witness[v]) {
            *slot = ring(e);
        }
    }
    Ok(VerticallyAlignedMatrix {
        data,
        width,
        height,
        used_cols: width,
    })
}

/// The tensor-structured Ajtai keys of `T_Y` and `T_u`, with their rows expanded.
pub struct Keys {
    /// The residue regions, then the lift region, over the layout's own length.
    pub residues: Vec<Region>,
    pub lift: Region,
    /// `rows_y[i][k]`: the `log2(residues[k].len())` layers of key row `i` over region `k`.
    pub rows_y: Vec<Vec<Vec<RingElement>>>,
    /// `rows_u[i]`: the layers of key row `i` over the lift region.
    pub rows_u: Vec<Vec<RingElement>>,
    /// `PreprocessedRow::from_layers` of every row, `region.len()` elements each.
    pub expanded_y: Vec<Vec<Vec<RingElement>>>,
    pub expanded_u: Vec<Vec<RingElement>>,
}

impl Keys {
    /// `residues`, `lift`: layout indices; `rank_y` rows then `rank_u` rows from `seed`.
    pub fn new(
        layout: &Layout,
        residues: &[usize],
        lift: usize,
        seed: [u8; 32],
        rank_y: usize,
        rank_u: usize,
    ) -> Keys {
        let mut sampler = AesCtrPublicSampler::from_seed(&seed);
        let mut layers = |len: usize| -> Vec<RingElement> {
            (0..len.ilog2())
                .map(|_| {
                    let mut e = zero();
                    sampler.fill_ring_element(&mut e, Representation::IncompleteNTT);
                    e
                })
                .collect()
        };
        let residues: Vec<Region> = residues
            .iter()
            .map(|&v| region(layout, v, layout.len))
            .collect();
        let lift = region(layout, lift, layout.len);
        let rows_y: Vec<Vec<Vec<RingElement>>> = (0..rank_y)
            .map(|_| residues.iter().map(|r| layers(r.len())).collect())
            .collect();
        let rows_u: Vec<Vec<RingElement>> = (0..rank_u).map(|_| layers(lift.len())).collect();
        let expand =
            |layers: &Vec<RingElement>| PreprocessedRow::from_layers(layers).preprocessed_row;
        let expanded_y = rows_y
            .iter()
            .map(|row| row.iter().map(expand).collect())
            .collect();
        let expanded_u = rows_u.iter().map(expand).collect();
        Keys {
            residues,
            lift,
            rows_y,
            rows_u,
            expanded_y,
            expanded_u,
        }
    }
}

fn inner(key: &[RingElement], vector: &[Element]) -> RingElement {
    assert!(
        vector.len() <= key.len(),
        "{} elements under a key of {}",
        vector.len(),
        key.len()
    );
    let mut acc = zero();
    let mut term = zero();
    for (k, e) in key.iter().zip(vector) {
        let e = ring(e);
        term *= (k, &e);
        acc += &term;
    }
    acc
}

/// `vectors[k]`: the `used` elements of the vector at `keys.residues[k]`.
pub fn commit_residues(keys: &Keys, vectors: &[Vec<Element>]) -> Vec<RingElement> {
    assert_eq!(
        vectors.len(),
        keys.residues.len(),
        "one vector per residue region"
    );
    keys.expanded_y
        .iter()
        .map(|row| {
            let mut acc = zero();
            for (key, vector) in row.iter().zip(vectors) {
                acc += &inner(key, vector);
            }
            acc
        })
        .collect()
}

pub fn commit_lift(keys: &Keys, lift: &[Element]) -> Vec<RingElement> {
    keys.expanded_u.iter().map(|key| inner(key, lift)).collect()
}

/// What the claims need of the key alone: the NTT form of the fixed weight polynomials
/// (`Relation::polys[..fixed]`) and `conj(sum_{j < SUPPORT} X^j)` of the binariness claim.
pub struct Fixed {
    pub ntt: Vec<RingElement>,
    pub ones_conjugate: RingElement,
}

impl Fixed {
    pub fn new(polys: &[Poly]) -> Fixed {
        Fixed {
            ntt: polys.iter().map(weight).collect(),
            ones_conjugate: weight(&vec![1; SUPPORT]).conjugate(),
        }
    }
}

/// Rokoko's reference string for one chain, shared by every proof under it.
pub struct Crs {
    pub prover: CRS,
    pub verifier: VerifierCRS,
}

impl Crs {
    pub fn new(config: &SumcheckConfig) -> Crs {
        Crs {
            prover: CRS::gen_prover_crs(config),
            verifier: CRS::gen_verifier_crs(config),
        }
    }
}

pub struct Proof {
    pub commitment: Vec<RingElement>,
    /// `V_k = sum_i alpha_{k,i} w_i`, coefficients `[SUPPORT, DEG)` zero.
    pub support: Vec<RingElement>,
    /// Per vector, `sum w conj(w)`: constant term the squared `l2` norm.
    pub norms: Vec<RingElement>,
    /// Per binary vector, `sum w conj(w) - conj(J) w`: constant term zero.
    pub binary: Vec<RingElement>,
    pub claims: ClaimsProof,
    pub chain: SumcheckRoundProof,
}

impl Proof {
    fn shipped(&self) -> impl Iterator<Item = &RingElement> {
        self.commitment
            .iter()
            .chain(&self.support)
            .chain(&self.norms)
            .chain(&self.binary)
    }

    pub fn wire_bytes(&self) -> usize {
        let elements: usize = self.shipped().map(RingElement::compact_size_in_bits).sum();
        (elements + self.claims.size_in_bits() + self.chain.size_in_bits()).div_ceil(8)
    }
}

fn transcript(
    digest: [u8; 32],
    t_y: &[RingElement],
    t_u: &[RingElement],
    commitment: &[RingElement],
) -> HashWrapper {
    let mut h = HashWrapper::new();
    h.update_with_bytes(&digest);
    h.update_with_ring_element_slice(t_y);
    h.update_with_ring_element_slice(t_u);
    h.update_with_ring_element_slice(commitment);
    h
}

fn powers(x: &RingElement, count: usize) -> Vec<RingElement> {
    let mut p = Vec::with_capacity(count);
    let mut acc = one();
    for _ in 0..count {
        p.push(acc.clone());
        acc *= x;
    }
    p
}

fn sum(terms: impl IntoIterator<Item = ClaimExpr>) -> Option<ClaimExpr> {
    terms.into_iter().reduce(|a, b| a + b)
}

/// `acc += s * x`, lane-wise: a scalar acts on every NTT slot alike.
fn fma(acc: &mut RingElement, x: &RingElement, s: u64) {
    unsafe {
        eltwise_fma_mod(
            acc.v.as_mut_ptr(),
            x.v.as_ptr(),
            s,
            acc.v.as_ptr(),
            DEG as u64,
            MOD_Q,
        )
    }
}

fn residue(s: i64) -> u64 {
    s.rem_euclid(MOD_Q as i64) as u64
}

/// `sum w conj(w)` over the elements.
fn energy(elements: &[RingElement]) -> RingElement {
    let mut acc = zero();
    let mut term = zero();
    let mut conj = zero();
    for w in elements {
        w.conjugate_into(&mut conj);
        term *= (w, &conj);
        acc += &term;
    }
    acc
}

/// Products `rho^a * poly` kept across the chains of one round, at most this many (a ring
/// element each).
const SCALED: usize = 1 << 16;

/// A claim whose value the prover ships.
pub enum Shipped<'a> {
    Support(&'a [u64]),
    Norm(Region),
    Binary(Region),
}

fn support_weights(owner: &[Option<usize>], n: usize, transcript: &mut HashWrapper) -> Vec<u64> {
    (0..n)
        .map(|i| match owner.get(i) {
            Some(Some(_)) => transcript.sample_u64_mod_q(),
            _ => 0,
        })
        .collect()
}

/// `ship` values the witness-dependent claims: the prover sums, the verifier reads the proof.
fn claims(
    relation: &Relation,
    keys: &Keys,
    fixed: &Fixed,
    t_y: &[RingElement],
    t_u: &[RingElement],
    n: usize,
    transcript: &mut HashWrapper,
    ship: &mut dyn FnMut(Shipped) -> RingElement,
) -> Result<Vec<Claim>, String> {
    let layout = &relation.layout;
    let owner = owners(layout);
    let at = |v: usize| region(layout, v, n);
    let mut claims = Vec::new();

    if relation.fixed != fixed.ntt.len() || relation.polys.len() < relation.fixed {
        return Err(format!(
            "the relation fixes {} of {} polynomials, the key holds {}",
            relation.fixed,
            relation.polys.len(),
            fixed.ntt.len()
        ));
    }
    let clock = Instant::now();
    let round: Vec<RingElement> = relation.polys[relation.fixed..]
        .iter()
        .map(weight)
        .collect();
    stamp(
        &format!("round polys ({})", relation.polys.len() - relation.fixed),
        clock,
    );
    let clock = Instant::now();
    let ntt = |p: usize| -> Result<&RingElement, String> {
        match fixed.ntt.get(p) {
            Some(e) => Ok(e),
            None => round
                .get(p - relation.fixed)
                .ok_or_else(|| format!("entry weight {p} is not among the polynomials")),
        }
    };

    let mut rho = zero();
    transcript.sample_ring_element_into(&mut rho);
    let mut entries = 0;
    let diagonals = relation.equations.iter().map(|c| c.diagonals.len()).max();
    let rho = powers(&rho, diagonals.unwrap_or(0));
    let mut slot = vec![u32::MAX; relation.polys.len()];
    let mut scaled: Vec<(usize, RingElement)> =
        Vec::with_capacity(SCALED.min(relation.polys.len()));
    for chain in &relation.equations {
        let mut tables: Vec<Option<Vec<RingElement>>> = vec![None; layout.vectors.len()];
        let mut value = zero();
        let mut term = zero();
        for (a, diagonal) in chain.diagonals.iter().enumerate() {
            entries += diagonal.entries.len();
            for entry in &diagonal.entries {
                let i = entry.element;
                let v = owner
                    .get(i)
                    .copied()
                    .flatten()
                    .ok_or_else(|| format!("chain {} reads pad element {i}", chain.name))?;
                let table = tables[v].get_or_insert_with(|| vec![zero(); layout.regions[v].len]);
                let cached = match slot[entry.poly] as usize {
                    k if k < scaled.len() && scaled[k].0 == a => Some(k),
                    _ if scaled.len() < SCALED => {
                        let k = scaled.len();
                        scaled.push((a, zero()));
                        scaled[k].1 *= (&rho[a], ntt(entry.poly)?);
                        slot[entry.poly] = k as u32;
                        Some(k)
                    }
                    _ => None,
                };
                let weight = match cached {
                    Some(k) => &scaled[k].1,
                    None => {
                        term *= (&rho[a], ntt(entry.poly)?);
                        &term
                    }
                };
                fma(
                    &mut table[i - layout.regions[v].start],
                    weight,
                    residue(entry.scale),
                );
            }
            term *= (&rho[a], &weight(&diagonal.output));
            value += &term;
        }
        let terms = tables
            .into_iter()
            .enumerate()
            .filter_map(|(v, t)| t.map(|t| table(t).on(at(v).vars()) * witness_in(at(v))));
        let expr = sum(terms).ok_or_else(|| format!("chain {} reads nothing", chain.name))?;
        claims.push(Claim::sums_to(expr, value));
    }
    stamp(
        &format!(
            "tables ({entries} entries, {} polys, {} fixed)",
            relation.polys.len(),
            relation.fixed
        ),
        clock,
    );

    if t_y.len() != keys.rows_y.len() || t_u.len() != keys.rows_u.len() {
        return Err(format!(
            "T_Y has {} rows for a key of {}, T_u {} for {}",
            t_y.len(),
            keys.rows_y.len(),
            t_u.len(),
            keys.rows_u.len()
        ));
    }
    let over = |r: &Region| Region::new(r.start(), r.len(), n);
    for (row, t) in keys.rows_y.iter().zip(t_y) {
        let terms = row.iter().zip(&keys.residues).map(|(layers, r)| {
            let r = over(r);
            eq(layers.clone()).on(r.vars()) * witness_in(r)
        });
        let expr = sum(terms).ok_or("T_Y covers no region")?;
        claims.push(Claim::sums_to(expr, t.clone()));
    }
    let lift = over(&keys.lift);
    for (layers, t) in keys.rows_u.iter().zip(t_u) {
        claims.push(Claim::sums_to(
            eq(layers.clone()).on(lift.vars()) * witness_in(lift),
            t.clone(),
        ));
    }

    for _ in 0..SUPPORT_CLAIMS {
        let alpha = support_weights(&owner, n, transcript);
        let value = ship(Shipped::Support(&alpha));
        claims.push(Claim::sums_to(table(alpha) * witness(), value));
    }

    for v in 0..layout.vectors.len() {
        let r = at(v);
        let value = ship(Shipped::Norm(r));
        claims.push(Claim::sums_to(
            witness_in(r) * witness_in(r).conjugate(),
            value,
        ));
    }

    let j = &fixed.ones_conjugate;
    for v in (0..layout.vectors.len()).filter(|&v| layout.vectors[v].binary) {
        let r = at(v);
        let value = ship(Shipped::Binary(r));
        claims.push(Claim::sums_to(
            witness_in(r) * witness_in(r).conjugate()
                - table(vec![j.clone(); r.len()]).on(r.vars()) * witness_in(r),
            value,
        ));
    }

    Ok(claims)
}

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(rokoko::common::init_common);
}

pub fn prove(
    relation: &Relation,
    witness: &Witness,
    keys: &Keys,
    fixed: &Fixed,
    t_y: &[RingElement],
    t_u: &[RingElement],
    crs: &Crs,
    config: &SumcheckConfig,
    digest: [u8; 32],
) -> Result<Proof, String> {
    init();
    ntt(t_y.iter().chain(t_u))?;
    let layout = &relation.layout;
    let committed = committed(layout, witness, config)?;
    let n = committed.data.len();

    let crs = &crs.prover;
    let clock = Instant::now();
    let mut context = init_sumcheck(crs, config);
    stamp("init_sumcheck", clock);
    let clock = Instant::now();
    let (commitment, root) = commit(crs, config, &committed);
    stamp("commit", clock);

    let mut transcript = transcript(digest, t_y, t_u, &root);
    let mut shipped = Vec::new();
    let data = &committed.data;
    let clock = Instant::now();
    let claims = claims(
        relation,
        keys,
        fixed,
        t_y,
        t_u,
        n,
        &mut transcript,
        &mut |claim| {
            let value = match claim {
                Shipped::Support(alpha) => {
                    let mut acc = zero();
                    for (w, &a) in data.iter().zip(alpha).filter(|(_, &a)| a != 0) {
                        fma(&mut acc, w, a);
                    }
                    acc
                }
                Shipped::Norm(r) => energy(&data[r.range()]),
                Shipped::Binary(r) => {
                    let mut total = zero();
                    for w in &data[r.range()] {
                        total += w;
                    }
                    let mut value = energy(&data[r.range()]);
                    let mut term = zero();
                    term *= (&fixed.ones_conjugate, &total);
                    value -= &term;
                    value
                }
            };
            shipped.push(value.clone());
            value
        },
    )?;
    stamp("prover claims", clock);
    let clock = Instant::now();
    let (claims, inputs) = prove_claims(&committed, &claims, &mut transcript);
    stamp("prove_claims", clock);
    let clock = Instant::now();
    let (chain, _) = prover_round(
        crs,
        config,
        &commitment,
        &committed,
        &inputs.evaluation_points_inner,
        &inputs.evaluation_points_outer,
        &mut context,
        false,
        Some(transcript),
        None,
    );
    stamp("prover_round", clock);

    let binary = shipped.split_off(SUPPORT_CLAIMS + layout.vectors.len());
    let norms = shipped.split_off(SUPPORT_CLAIMS);
    Ok(Proof {
        commitment: root,
        support: shipped,
        norms,
        binary,
        claims,
        chain,
    })
}

fn cap_squared(cap: Cap, used: usize) -> f64 {
    match cap {
        Cap::Betasq(b) => b,
        Cap::PerCoefficient(c) => c * c * (used * SUPPORT) as f64,
    }
}

fn structure(relation: &Relation, proof: &Proof) -> Result<(), String> {
    let layout = &relation.layout;
    let binary = layout.vectors.iter().filter(|v| v.binary).count();
    if proof.support.len() != SUPPORT_CLAIMS
        || proof.norms.len() != layout.vectors.len()
        || proof.binary.len() != binary
    {
        return Err("the proof ships the wrong number of values".into());
    }
    for (k, v) in proof.support.iter().enumerate() {
        let c = coefficients(v);
        if let Some(j) = (SUPPORT..DEG).find(|&j| c[j] != 0) {
            return Err(format!("support claim {k}: coefficient {j} is {}", c[j]));
        }
    }
    for (v, (vector, value)) in layout.vectors.iter().zip(&proof.norms).enumerate() {
        let norm = constant_term(value);
        let cap = cap_squared(vector.cap, vector.used);
        if norm < 0 || norm as f64 > cap {
            return Err(format!(
                "norm of vector {v} ({}) is {norm}, cap {cap}",
                vector.name
            ));
        }
    }
    for (vector, value) in layout
        .vectors
        .iter()
        .filter(|v| v.binary)
        .zip(&proof.binary)
    {
        let sum = constant_term(value);
        if sum != 0 {
            return Err(format!(
                "vector {} is not binary: sum x(x - 1) = {sum}",
                vector.name
            ));
        }
    }
    Ok(())
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "rokoko rejected the proof".into()
    }
}

/// Rokoko rejects by panicking; the panic is caught and returned as the error.
pub fn verify(
    relation: &Relation,
    keys: &Keys,
    fixed: &Fixed,
    t_y: &[RingElement],
    t_u: &[RingElement],
    crs: &Crs,
    config: &SumcheckConfig,
    digest: [u8; 32],
    proof: &Proof,
) -> Result<(), String> {
    init();
    ntt(t_y.iter().chain(t_u).chain(proof.shipped()))?;
    let layout = &relation.layout;
    let (height, width) = shape(layout, config)?;
    let n = height * width;
    structure(relation, proof)?;
    let mut transcript = transcript(digest, t_y, t_u, &proof.commitment);
    let mut shipped = proof
        .support
        .iter()
        .chain(&proof.norms)
        .chain(&proof.binary);
    let clock = Instant::now();
    let claims = claims(
        relation,
        keys,
        fixed,
        t_y,
        t_u,
        n,
        &mut transcript,
        &mut |_| shipped.next().cloned().expect("counted by structure"),
    )?;
    stamp("claims", clock);
    let clock = Instant::now();
    let bound = super::relation::no_wrap_bound(relation);
    stamp("no_wrap_bound", clock);
    if !(bound < (MOD_Q / 2) as f64) {
        return Err(format!(
            "the block equations may wrap: bound {bound} against q/2 = {}",
            MOD_Q / 2
        ));
    }

    let clock = Instant::now();
    let mut context = init_verifier(&crs.verifier, config);
    stamp("init_verifier", clock);

    catch_unwind(AssertUnwindSafe(|| {
        let clock = Instant::now();
        let inputs = verify_claims(
            WitnessShape::new(height, width),
            &claims,
            &proof.claims,
            &mut transcript,
        );
        stamp("verify_claims", clock);
        let clock = Instant::now();
        verifier_round(
            &crs.verifier,
            config,
            &proof.commitment,
            &proof.chain,
            &inputs.evaluation_points_inner,
            &inputs.evaluation_points_outer,
            &inputs.claims,
            &mut context,
            Some(transcript),
            None,
        );
        stamp("verifier_round", clock);
    }))
    .map_err(panic_message)
}
