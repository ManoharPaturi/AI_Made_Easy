# AI Made Easy 3.4 — plan and status

*Status: in progress. Phases 1 and 2 (everything 3.3 moved here) are done; the rest of
3.4 is still to be planned with you.*

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

## Phase 2 — the rest of the deferred list (`feat/v34-phase2`)
- **Cosine Embedding and Margin Ranking losses** (`train.loss_cosine_embedding`,
  `train.loss_margin_ranking`). **Done.**
  - *Cosine embedding:* every same-class pair in a batch pulls its cosine towards 1, and
    every other pair pushes it below a margin.
  - *Margin ranking:* around each example, every same-class neighbour must be more
    similar than every other-class one, by the margin.
  - **Deviation:** the plan was a pairs-dataset block first. Instead, the pairs are made
    from the class labels in each batch, the way the batch-hard triplet loss already
    works. That way the losses work with every dataset that has classes, and they score
    accuracy the same way as the other metric losses. A dataset of explicit labelled
    pairs (e.g. two columns of ids and a same / different label) stays open, for when
    someone needs it.
- **Wires on folded desktop groups.** Wires that cross a folded group's border are drawn
  as dashed stand-ins from the card's edge to the block outside (or to another folded
  card), and they follow the card. Only the drawing changes: the real wires are kept, so
  the design and its code don't change. This replaces the proxy ports the plan had; the
  result looks the same and the canvas library isn't touched. **Done.**

## Still open
- A dataset block of explicit labelled pairs (see the deviation above).
- The next round of 3.4 features, to be agreed.
