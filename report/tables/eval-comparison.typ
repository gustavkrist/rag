#let data = csv("../../eval/eval-comparison-table-qwen3-1.7B.csv")
#set table(stroke: (x, y) => if x == 0 and y == 0 { none })
#let format-rows(rows, skip: 1) = {
  rows
    .map(row => {
      let max = calc.max(..row.slice(skip).map(float))

      row
        .enumerate()
        .map(((i, v)) => {
          if i >= skip and float(v) == max { [*#v*] } else { [#v] }
        })
    })
    .flatten()
}
#table(
  columns: 3,
  table.header(..data.at(0)),
  ..format-rows(data.slice(1)),
)
