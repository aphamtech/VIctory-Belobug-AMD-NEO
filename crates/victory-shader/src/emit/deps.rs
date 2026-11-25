//! The input-output dependence masks `PSV0` and the view-id state carry:
//! which signature output scalars each signature input scalar reaches.
//!
//! `PSV0` is checked against this by the DXIL validator, byte for byte:
//! declaring every output dependent on every input (what this used to do)
//! only holds where the shader's dataflow happens to reach that far, and
//! the validator rejects the rest (`Container part 'Pipeline State
//! Validation' does not match expected for module`). The dependence is read
//! off the emitted module: from each `dx.op.storeOutput` back through the
//! values (per-lane temporaries included) to the `dx.op.loadInput` calls
//! that feed it.

use std::collections::{HashMap, HashSet};

use super::bitcode::{Module, Op, V};
use super::container::SigElement;

/// One dword mask per input scalar (four per input row), each bit one
/// output scalar.
pub(crate) fn masks(m: &Module, inputs: &[SigElement], outputs: &[SigElement]) -> Vec<u32> {
    let rows = |els: &[SigElement]| {
        els.iter()
            .filter_map(|e| e.row)
            .max()
            .map_or(0, |r| r as usize + 1)
    };
    let ni = rows(inputs) * 4;
    let no = rows(outputs) * 4;
    if ni == 0 || no == 0 {
        return Vec::new();
    }
    let per = no.div_ceil(32);
    let mut out = vec![0u32; ni * per];
    // Values written into each lane-local temporary: a load from one takes
    // what was stored there, and the shader's dataflow runs through them.
    let stored = stored_values(m);
    for v in m.instruction_ids() {
        let Some(Op::Call(f, args)) = m.op_of(v) else {
            continue;
        };
        if !m.func_name(*f).starts_with("dx.op.storeOutput") {
            continue;
        }
        // storeOutput(i32 code, i32 element, i32 0, i8 component, value)
        let (Some(id), Some(component), Some(value)) = (
            args.get(1).and_then(|&a| m.known(a)),
            args.get(3).and_then(|&a| m.known(a)),
            args.last(),
        ) else {
            continue;
        };
        let Some(target) = scalar(outputs, id as usize, component as u8) else {
            continue;
        };
        if target >= no {
            continue;
        }
        let mut reached = HashSet::new();
        walks(m, &stored, *value, &mut reached);
        for (id, component) in reached {
            let Some(at) = scalar(inputs, id, component) else {
                continue;
            };
            if at < ni {
                out[at * per + target / 32] |= 1 << (target % 32);
            }
        }
    }
    out
}

/// Which signature scalar an element's component is: four per row.
fn scalar(els: &[SigElement], id: usize, component: u8) -> Option<usize> {
    let e = els.get(id)?;
    let row = e.row? as usize;
    Some(row * 4 + e.col as usize + component as usize)
}

/// Every value stored into each per-lane temporary (`alloca`), whichever
/// element of it: this is not an alias analysis, and taking them all is the
/// safe side of declaring a dependence.
fn stored_values(m: &Module) -> HashMap<u32, Vec<V>> {
    let mut stored: HashMap<u32, Vec<V>> = HashMap::new();
    for v in m.instruction_ids() {
        let Some(op) = m.op_of(v) else { continue };
        let (ptr, value) = match op {
            Op::Store(ptr, value) => (*ptr, *value),
            Op::AtomicRmw(_, ptr, value) => (*ptr, *value),
            Op::CmpXchg(ptr, _, value) => (*ptr, *value),
            _ => continue,
        };
        if let Some(alloca) = alloca_of(m, ptr) {
            stored.entry(alloca).or_default().push(value);
        }
    }
    stored
}

/// The instruction of the `alloca` a pointer comes from, if it is one.
fn alloca_of(m: &Module, mut v: V) -> Option<u32> {
    // A chain of operands, not a loop.
    for _ in 0..32 {
        let i = match v {
            V::Inst(i) => i,
            _ => return None,
        };
        match m.op_of(v)? {
            Op::Alloca(..) => return Some(i),
            Op::Gep(_, base, _) | Op::Cast(_, base) | Op::Extract(base, _) => v = *base,
            _ => return None,
        }
    }
    None
}

/// Walks the values behind `v`, collecting the `(element, component)` of
/// every `dx.op.loadInput` it reads.
fn walks(m: &Module, stored: &HashMap<u32, Vec<V>>, v: V, reached: &mut HashSet<(usize, u8)>) {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        let V::Inst(i) = v else { continue };
        if !seen.insert(i) {
            continue;
        }
        let Some(op) = m.op_of(v) else { continue };
        match op {
            Op::Call(f, args) => {
                if m.func_name(*f).starts_with("dx.op.loadInput") {
                    // loadInput(i32 code, i32 element, i32 0, i8 component, i32 vertex)
                    if let (Some(id), Some(component)) = (
                        args.get(1).and_then(|&a| m.known(a)),
                        args.get(3).and_then(|&a| m.known(a)),
                    ) {
                        reached.insert((id as usize, component as u8));
                    }
                    continue;
                }
                stack.extend(args.iter().copied());
            }
            Op::Bin(_, a, b) | Op::Cmp(_, a, b) => stack.extend([*a, *b]),
            Op::Cast(_, a) | Op::Extract(a, _) => stack.push(*a),
            Op::Load(ptr) => {
                stack.push(*ptr);
                if let Some(alloca) = alloca_of(m, *ptr)
                    && let Some(values) = stored.get(&alloca)
                {
                    stack.extend(values.iter().copied());
                }
            }
            Op::AtomicRmw(_, ptr, value) => stack.extend([*ptr, *value]),
            Op::Select(c, t, f) | Op::CmpXchg(c, t, f) => stack.extend([*c, *t, *f]),
            Op::Phi(incoming) => stack.extend(incoming.iter().map(|&(v, _)| v)),
            Op::Gep(_, base, idx) => {
                stack.push(*base);
                stack.extend(idx.iter().copied());
            }
            Op::Store(..) | Op::Alloca(..) => {}
        }
    }
}
