# RAW: native arithmetic fit versus transfer

LMM source: anjaustin/lmm at 55719c0a2308d1a3f29ca656ed7b6287be33bca2.

Observed: task-only arithmetic still misses 51/128 training compound answers;
primitive multiplication and subtraction fit perfectly. Ordering fits 320/320
but changes from 10/20 bundled to 18/20 specialist on exposed diagnostics.
More data or another wording could help, but repeated intervention on the same
100 cases risks benchmark chasing. We have not varied training duration or
adapter rank in isolated arithmetic. One epoch is a budget choice, not a floor.
Worked answers and final-only answers have different formats and could interact.
Changing multiple mechanisms again would obscure why.

Open questions: insufficient exposures, rank, operator target placement,
mixed worked/final supervision, optimizer/clipping, numerical inference mismatch,
or inability to learn a compositional rule? No single explanation is proven.
