# Humanizer audit: technical report

## Draft diagnosis

The first LaTeX draft was technically accurate, but several passages still read
like generated academic prose:

- The abstract opened with a generic “We describe” construction and compressed
  too many method nouns into one sentence.
- Some transitions announced their purpose instead of making the point
  directly, especially “These failures changed the system in concrete ways.”
- Several paragraphs used polished three-part lists where a shorter statement
  would sound more natural.
- “Our main contribution” and “The strongest improvement came from” were
  defensible, but too close to stock paper language.
- The deployment section repeated “frozen” often enough to sound mechanical.
- A few claims were broad when the underlying experiment could be named
  instead.

## Revision standard

The final pass keeps the IEEE tone and all measured values, but:

- starts sections with the actual finding;
- replaces generic transitions with named experiments or decisions;
- varies sentence length without becoming conversational;
- keeps caveats next to the claims they qualify;
- avoids em dashes, en dashes, rhetorical questions, hype, and false quotes;
- uses plain technical language for compliance and reproducibility statements.

The final source is checked for common generated-writing markers before
publication.
