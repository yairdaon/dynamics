# Course TODO

## Module: State-space reconstruction (EDM)
- Native-coordinate attractor: simulate the SEIR model with the measles
  parameters Aron & Schwartz used (as reported in Olsen & Schaffer 1990):
  $m=0.02\,\text{yr}^{-1}$, $a=35.84\,\text{yr}^{-1}$, $g=100\,\text{yr}^{-1}$,
  $\beta_0=1800\,\text{yr}^{-1}$, $\beta_1=0.28$. Plot the attractor in
  native coordinates $S(t), I(t), \beta(t)$ and note it is diffeomorphic to
  Olsen & Schaffer's lagged-coordinate reconstruction
  $I(t), I(t+T), I(t+2T)$ (their Fig. 2A, $T=3$ months).
- Introduce S-maps on the non-reconstructed (native) state space
  ($S,I,\beta$, or any other reasonable native coordinates) as a first
  look at local/nonlinear structure before reconstruction is introduced.
- (Later lecture) Present S-maps as a state-space reconstruction
  technique proper (lagged coordinates) and recover the Jacobian of the
  system from data.
- (Later lecture) Convergent cross mapping (CCM).
