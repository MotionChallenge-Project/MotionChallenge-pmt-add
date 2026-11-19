# MotionChallenge · pmt-add

This checkpoint is excluded from the MotionChallenge protocol because its released policy observation consumes actual robot linear velocity.

The protocol excludes actual global position and actual linear velocity in every frame, including explicit estimates. Reference root/joint motion and their derivatives are permitted.

[Checkpoint admission](admission.mjwarp.json) · [Observation protocol](MotionChallenge-Core/docs/OBSERVATIONS.md)

The evaluator checks the admission record and rejects this checkpoint. No LAFAN or SEED leaderboard score is published for it.
