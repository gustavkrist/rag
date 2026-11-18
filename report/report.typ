#import "@preview/basic-report:0.5.0": *

#show: it => basic-report(
  // doc-category: "Lab Report",
  doc-title: "Assessment 2: LLM Project",
  author: "Gustav Kristensen",
  affiliation: "James Cook University Singapore",
  logo: image("images/logo.png", width: 2cm),
  language: "en",
  compact-mode: false,
  it,
)

#show table.cell.where(y: 0): strong

= Introduction

= Target Problem

= Data Sources

= Methodology

= Evaluation

#figure(
  include("tables/eval-stats.typ"),
  caption: "Evaluation Results",
) <eval-stats>

= Discussion

= Conclusion

#bibliography(
  "bibliography.bib",
  style: "american-psychological-association",
  full: true,
  title: "References",
)
