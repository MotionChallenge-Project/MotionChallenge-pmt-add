# MotionChallenge · pmt-add

This checkpoint is excluded from the MotionChallenge protocol because its released policy observation consumes actual robot linear velocity.

The protocol excludes actual global position and actual linear velocity in every frame, including explicit estimates. Reference root/joint motion and their derivatives are permitted.

[Checkpoint admission](admission.mjwarp.json) · [Observation protocol](MotionChallenge-Core/docs/OBSERVATIONS.md)

The evaluator checks the admission record and rejects this checkpoint. No LAFAN or SEED leaderboard score is published for it.

## Metric

MPKPE (Mean Per-Keypoint Position Error) measures the 3D Euclidean distance between actual and reference keypoint positions. The protocol defines these keypoints at robot body-frame origins: 14 points for G1. Errors are averaged over keypoints and scoring samples within each motion, then motions are weighted equally. Results report full-motion and first-failure-prefix scores with no alignment, XY alignment, and XY + yaw alignment. XY alignment removes pelvis horizontal translation while retaining height error. Leaderboard values are reported in millimetres (mm).
