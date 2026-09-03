//! binius64's reductions, driven from here instead of from `IOPProver::prove`.
//!
//! `IOPProver::prove` runs the reductions and then, in the same body, its own `B128`
//! ring-switching and the BaseFold opening of the Merkle oracle. The channel's oracle hooks sit
//! before and after that pair, never between the reductions and the witness claim, so the loop is
//! reproduced here and stopped exactly where the claim `w~(r) = s` on the packed non-public trace
//! appears — which is what the cross-field switch consumes. Everything it calls is binius64's own
//! public reduction provers, run unchanged, on binius64's own transcript as the channel.
use binius_compute::Allocator;
use binius_core::constraint_system::{ConstraintSystem, InoutSegment, Operand, ValueVec};
use binius_core::word::Word;
use binius_field::{PackedField, Rijndael8b as B8};
use binius_iop::channel::IOPVerifierChannel;
use binius_ip::channel::WordIPVerifierChannel;
use binius_ip::sumcheck::SumcheckOutput;
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_math::univariate::EvaluationDomain;
use binius_math::BinarySubspace;
use binius_prover::protocols::shift::{
    prove as prove_shift, KeyCollection, OperatorClaims, OperatorData, ShiftOutput,
};
use binius_prover::{
    protocols::binmul, protocols::bitand as and_reduction, ring_switch,
};
use binius_verifier::config::B128;
use binius_verifier::protocols::binmul::BinMulOutput;
use binius_verifier::protocols::bitand::AndCheckOutput;
use binius_verifier::protocols::zero;
use binius_verifier::reduction::{reduce_constraints, Instances};
use binius_verifier::{Error, IOPVerifier};

/// Milliseconds per reduction, at the granularity binius64's own phase spans use.
#[derive(Clone, Copy, Default)]
pub struct ProveTiming {
    pub bitand: f64,
    pub shift: f64,
}

/// The verifier's two halves: replaying the reductions, and discharging the wiring claim.
#[derive(Clone, Copy, Default)]
pub struct VerifyTiming {
    pub reduce: f64,
    pub wiring: f64,
}

fn milliseconds(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

/// The constraint system with everything the two sides precompute from it.
pub struct Liop {
    iop: IOPVerifier,
    keys: KeyCollection,
}

impl Liop {
    pub fn new(constraint_system: ConstraintSystem) -> Liop {
        constraint_system
            .validate()
            .expect("the circuit compiles to a valid system");
        assert_eq!(
            constraint_system.n_imul_constraints(),
            0,
            "the IntMul reduction is bounded on an IOP channel, which this pipeline has no oracle for"
        );
        let log_public_words = constraint_system.log_public_words(InoutSegment::Public);
        let keys = KeyCollection::build(&constraint_system, InoutSegment::Public);
        Liop {
            iop: IOPVerifier::new(constraint_system, log_public_words),
            keys,
        }
    }

    pub const fn log_witness_elems(&self) -> usize {
        self.iop.log_witness_elems()
    }

    pub const fn constraint_system(&self) -> &ConstraintSystem {
        self.iop.constraint_system()
    }

    /// The reductions, ending at the claim point on the packed trace. The claim's value is the
    /// prover's own and never sent; the verifier derives it from the same reductions.
    pub fn prove<A, P, Channel>(
        &self,
        witness: &ValueVec,
        channel: &mut Channel,
        alloc: &A,
        timing: &mut ProveTiming,
    ) -> Vec<B128>
    where
        A: Allocator,
        P: PackedField<Scalar = B128>,
        Channel: IPProverChannel<B128> + WordIPProverChannel<B128, Word = Word>,
    {
        let cs = self.constraint_system();
        let start = std::time::Instant::now();

        let binmul_output = (cs.n_bmul_constraints() > 0).then(|| {
            let c = columns::<_, 6, 6>(&cs.bmul_constraints, witness);
            let [a_lo, a_hi, b_lo, b_hi, c_lo, c_hi] = &c;
            binmul::prove::<_, _, P, _>(
                [a_lo, a_hi, b_lo, b_hi, c_lo, c_hi].map(Vec::as_slice),
                &mut *channel,
                alloc,
            )
        });

        let bitand_claim = {
            let AndCheckOutput {
                a_eval,
                b_eval,
                c_eval,
                z_challenge,
                eval_point,
            } = and_reduction::prove::<_, B128, P, _, _>(
                columns::<_, 3, 2>(&cs.and_constraints, witness),
                &mut *channel,
                alloc,
            );
            OperatorData {
                evals: [a_eval, b_eval, c_eval],
                r_zhat_prime: z_challenge,
                r_x_prime: eval_point,
            }
        };

        timing.bitand = milliseconds(start);

        let subspace = BinarySubspace::<B8>::with_dim(Word::LOG_BITS).isomorphic();
        let binmul_claim = match binmul_output {
            Some(BinMulOutput {
                eval_point,
                a_lo_evals,
                a_hi_evals,
                b_lo_evals,
                b_hi_evals,
                c_lo_evals,
                c_hi_evals,
            }) => {
                let r_zhat_prime = bitand_claim.r_zhat_prime;
                let l_tilde = subspace.lagrange_evals_buffer(r_zhat_prime);
                let collapse = |evals| {
                    binius_math::inner_product::inner_product(evals, l_tilde.iter_scalars())
                };
                OperatorData {
                    evals: [
                        collapse(a_lo_evals),
                        collapse(a_hi_evals),
                        collapse(b_lo_evals),
                        collapse(b_hi_evals),
                        collapse(c_lo_evals),
                        collapse(c_hi_evals),
                    ],
                    r_zhat_prime,
                    r_x_prime: eval_point,
                }
            }
            None => OperatorData::zero_claim(bitand_claim.r_zhat_prime),
        };

        let log_n_zero = cs.log_zero_constraints().unwrap_or(0);
        let zero_claim = OperatorData {
            evals: [B128::default()],
            r_zhat_prime: bitand_claim.r_zhat_prime,
            r_x_prime: zero::reduction_point(&bitand_claim.r_x_prime, log_n_zero, || {
                channel.sample()
            }),
        };

        let start = std::time::Instant::now();
        let ShiftOutput {
            sumcheck:
                SumcheckOutput {
                    challenges: eval_point,
                    eval: _,
                },
            wiring_eval,
        } = prove_shift::<_, P, _, _>(
            &self.keys,
            witness.public(),
            witness.non_public(),
            OperatorClaims {
                zero: zero_claim,
                bitand: bitand_claim,
                intmul: OperatorData::zero_claim(B128::default()),
                binmul: binmul_claim,
            },
            &subspace,
            &mut *channel,
            alloc,
        );

        timing.shift = milliseconds(start);

        let witness_point = &eval_point[..eval_point.len() - 1];
        let (r_j, r_y) = witness_point.split_at(Word::LOG_BITS);
        ring_switch::prove_public_eval::<_, P, _>(alloc, witness.public(), r_j, r_y, &mut *channel);
        channel.send_public_claim(wiring_eval);

        witness_point.to_vec()
    }

    /// The same reductions on the verifier's side: the claim point and its value.
    pub fn verify<Channel>(
        &self,
        inout: &[Word],
        channel: &mut Channel,
    ) -> Result<(Vec<B128>, B128, VerifyTiming), Error>
    where
        Channel: IOPVerifierChannel<B128, Elem = B128> + WordIPVerifierChannel<B128, Word = Word>,
    {
        let cs = self.constraint_system();
        if inout.len() != cs.n_inout {
            return Err(Error::IncorrectPublicInputLength {
                expected: cs.n_inout,
                actual: inout.len(),
            });
        }
        let start = std::time::Instant::now();
        let public: Vec<Word> = cs
            .constants
            .iter()
            .copied()
            .chain(inout.iter().copied())
            .collect();
        let reduction = reduce_constraints(
            cs,
            Instances::Single,
            InoutSegment::Public,
            &public,
            channel,
        )?;
        let eval_point = reduction.trace_point();
        let claim = *reduction.shift.witness_eval();
        let reduce = milliseconds(start);

        let start = std::time::Instant::now();
        reduction.wiring.check_native()?;
        Ok((
            eval_point,
            claim,
            VerifyTiming {
                reduce,
                wiring: milliseconds(start),
            },
        ))
    }
}

/// Operands `0..N` of every constraint, one column each — `build_operation_columns` of
/// `binius_prover::prove`, which is private there.
fn columns<C, const ARITY: usize, const N: usize>(
    constraints: &[C],
    witness: &ValueVec,
) -> [Vec<Word>; N]
where
    C: AsRef<[Operand; ARITY]>,
{
    core::array::from_fn(|operand| {
        if constraints.is_empty() {
            return vec![Word::ZERO];
        }
        constraints
            .iter()
            .map(|c| witness.eval_operand(&c.as_ref()[operand]))
            .collect()
    })
}
