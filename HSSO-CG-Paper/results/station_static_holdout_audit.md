# Station-static holdout audit

The original receiver terms came from a coupled relocation/tomography workflow
containing 1,729 events. A direct identifier audit found that 28 of the 30
evaluation events contributed 219 picks to that workflow. Those terms were
therefore not suitable for a strictly held-out validation.

The revised terms are estimated against the fixed FSM velocity model after all
30 evaluation identifiers are excluded. The training set contains 1,701 events.
For every event, the median origin-time shift is removed before residuals are
grouped by receiver. The station term is the median finite residual, centred
across supported stations for identifiability.

A minimum of 20 finite training events is required. VMM05 (9 events), VMM11
(16), and VMM12 (2) receive zero corrections rather than unstable estimates.
The resulting terms and their median absolute deviations are stored in
`../../HSSO-Hypo/data/sgc_new_30/heldout_station_statics.csv`.

The 30 evaluation events are also absent from the 913-event inversion used to
construct the fixed P-wave model. Fifteen of the 32 stratified events overlap
that velocity-inversion set; the manuscript now identifies those 32 events as
an application set rather than independent validation.
