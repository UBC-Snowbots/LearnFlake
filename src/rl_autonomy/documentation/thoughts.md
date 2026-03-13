I don't think we should use stable baselines 3 because its too complicated and isn't compatible with HRL since it just has flat RL policies. 

Can use SB3 for environment validation (verifying whether or not the reason why the arm can/cannot pick up a block is either due to the environment or due to the actual model)