# AI Made Easy 3.4 — plan and status

*Status: in progress. Phase 1 (the items 3.3 moved here) is done; the rest of 3.4 is
still to be planned with you.*

## Phase 1 — what 3.3 deferred (`feat/v34-phase1`)
- **SHAP values** (`eval.shap_values`, `explain` extra). Expected gradients
  (`shap.GradientExplainer`) against training rows. It writes the same `explain.json` as
  permutation importance and Integrated Gradients, and both apps show it with the run.
  **Done.**
- **Collapsing groups.** A group folds into one card and opens again where the card was
  moved. Its blocks, their parameters and the generated code don't change.
  `meta.groups[].collapsed` is saved with the design. **Done.**
  - **Web:** wires to hidden blocks are re-routed to the card.
  - **Desktop:** the card is a backdrop of at least 230 × 106, and it remembers its blocks
    because a backdrop otherwise finds them by what it covers. Wires to folded blocks are
    hidden.
- **SimCLR.** The `train.loss_nt_xent` loss and an image pipeline that yields two
  augmentations of each training and validation image. The loop folds each pair into one
  batch, and each view's positive is its twin. The test set uses single views. The
  `image_simclr` recipe uses a CNN encoder with a projection head. **Done.**

## Still open
- **Cosine embedding and margin ranking losses.** These need datasets of pairs (or ranked
  pairs) with their own labels: same / different, or which item ranks higher. The
  two-view pipeline above makes positive pairs from one image, but not labelled pairs from
  a table. They need a pairs dataset block first.
- **Wires on folded desktop groups.** These are hidden rather than re-routed to the card,
  as the web does. OdenGraphQt draws pipes between ports, so re-routing needs proxy ports
  on the card.
- The next round of 3.4 features, to be agreed.
