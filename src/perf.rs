//! Minimal perf_event_open wrapper: one group of counters (cycles, instructions, and raw uop
//! events for Tiger Lake: uops_issued.any, uops_dispatched.port_0, port_1, port_5). Used by the
//! bench binary to report cycles and instruction/uop counts per polynomial. Linux only.
use std::io;

#[repr(C)]
struct PerfEventAttr {
    type_: u32,
    size: u32,
    config: u64,
    sample_period: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup_events: u32,
    bp_type: u32,
    bp_addr: u64,
    bp_len: u64,
    branch_sample_type: u64,
    sample_regs_user: u64,
    sample_stack_user: u32,
    clockid: i32,
    sample_regs_intr: u64,
    aux_watermark: u32,
    sample_max_stack: u16,
    reserved_2: u16,
    aux_sample_size: u32,
    reserved_3: u32,
    sig_data: u64,
    config3: u64,
}

extern "C" {
    fn syscall(num: i64, ...) -> i64;
    fn ioctl(fd: i32, req: u64, ...) -> i32;
    fn read(fd: i32, buf: *mut u8, n: usize) -> isize;
    fn close(fd: i32) -> i32;
}

const SYS_PERF_EVENT_OPEN: i64 = 298;
const PERF_TYPE_HARDWARE: u32 = 0;
const PERF_TYPE_RAW: u32 = 4;
const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
const PERF_FORMAT_GROUP: u64 = 1 << 3;
const PERF_IOC_RESET: u64 = 0x2403;
const PERF_IOC_ENABLE: u64 = 0x2400;
const PERF_IOC_DISABLE: u64 = 0x2401;
const PERF_IOC_FLAG_GROUP: u64 = 1;
// Tiger Lake / Ice Lake core PMU raw events (event | umask << 8).
const UOPS_ISSUED_ANY: u64 = 0x010e;
const UOPS_DISPATCHED_PORT_0: u64 = 0x01a1;
const UOPS_DISPATCHED_PORT_1: u64 = 0x02a1;
const UOPS_DISPATCHED_PORT_5: u64 = 0x20a1;

#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub cycles: u64,
    pub instructions: u64,
    pub uops: u64,
    pub port0: u64,
    pub port1: u64,
    pub port5: u64,
}

pub struct PerfGroup {
    fds: Vec<i32>,
}

impl PerfGroup {
    pub fn new() -> io::Result<Self> {
        let events: [(u32, u64); 6] = [
            (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES),
            (PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS),
            (PERF_TYPE_RAW, UOPS_ISSUED_ANY),
            (PERF_TYPE_RAW, UOPS_DISPATCHED_PORT_0),
            (PERF_TYPE_RAW, UOPS_DISPATCHED_PORT_1),
            (PERF_TYPE_RAW, UOPS_DISPATCHED_PORT_5),
        ];
        let mut fds = Vec::new();
        for (i, (ty, cfg)) in events.iter().enumerate() {
            let mut attr: PerfEventAttr = unsafe { std::mem::zeroed() };
            attr.type_ = *ty;
            attr.size = std::mem::size_of::<PerfEventAttr>() as u32;
            attr.config = *cfg;
            attr.read_format = PERF_FORMAT_GROUP;
            // disabled(1) | exclude_kernel(5) | exclude_hv(6)
            attr.flags = (if i == 0 { 1 } else { 0 }) | (1 << 5) | (1 << 6);
            let leader = if i == 0 { -1 } else { fds[0] };
            let fd = unsafe { syscall(SYS_PERF_EVENT_OPEN, &attr as *const _, 0i32, -1i32, leader, 0u64) };
            if fd < 0 {
                for f in &fds {
                    unsafe { close(*f) };
                }
                return Err(io::Error::last_os_error());
            }
            fds.push(fd as i32);
        }
        Ok(PerfGroup { fds })
    }
    pub fn start(&self) {
        unsafe {
            ioctl(self.fds[0], PERF_IOC_RESET, PERF_IOC_FLAG_GROUP);
            ioctl(self.fds[0], PERF_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        }
    }
    pub fn stop(&self) -> Counts {
        unsafe { ioctl(self.fds[0], PERF_IOC_DISABLE, PERF_IOC_FLAG_GROUP) };
        let mut buf = [0u64; 8];
        let n = unsafe { read(self.fds[0], buf.as_mut_ptr() as *mut u8, 8 * 8) };
        assert!(n >= 8 * 7, "short perf read");
        Counts {
            cycles: buf[1],
            instructions: buf[2],
            uops: buf[3],
            port0: buf[4],
            port1: buf[5],
            port5: buf[6],
        }
    }
}

impl Drop for PerfGroup {
    fn drop(&mut self) {
        for f in &self.fds {
            unsafe { close(*f) };
        }
    }
}
