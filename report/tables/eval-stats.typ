#let stats = csv("../../eval/eval-statistics-table-qwen3-1.7B.csv")
#table(
  columns: 8,
  table.header(..stats.at(0)),
  ..stats.slice(1).flatten().map(v => if v == "" { "NaN" } else { v }),
)
