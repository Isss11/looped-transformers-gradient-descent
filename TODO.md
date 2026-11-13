# TODO

## Additional Changes

- Display parameter graphs for looped models correctly, as well as the distance to identity graphs.

## Different Distributions Test

- Implement student's t-distribution with configurable parameters.
- Revise the training distribution to take appropriate parameters for gamma, t-distribution and pass them to the data generator.
- Revise the data generator to implement configurable gamma parameters for shape.
- Expand the simulations to include two H/L configurations for the gamma distribution and the t-distribution.
- Expand simulations to test on normal, gamma, and t-distribution data after training on the 5 types of distributions, and plot all 5 comparison graphs.
- Display parameter graphs.

## Non-Linear Distribution Test

- Implement on normal, ReLU generated data and MLP generated data.
- Test to see if convergence occurs.
- Display parameter graphs.

## Softmax Attention

- Implement softmax attention
- Test against regular attention, compare convergence graphs.
- Do we still see the phenomen after looping above the loops it was trained on? Test for this.
- Display parameter graphs.

## Conditional Number

- Verify implementation, see if conditional number is being set correctly.
- Compare performance with normal distribution.

## Unconstraining P & Q

- Look into unconstraining the bias term, as well as other terms if possible.
- Implement and test against regular models across different loops.
- Test looping for $L_{test} > L_{train}$.

## Multiple Heads

- Test with multiple heads, and see how that will effect results.
- Compare against single head models for $1, 2, 5, 10$ layers.

## $L_{test} > L_{train}$
- Investigate for even higher layers.
- Theoretical investigation.

## $\delta$ Test

- Investigate what $\delta$ really means, to see if this experiment is feasible.
- Modify values of $L$ and $N$ to see if they align with the estimation accuracy governed by $\delta$.

## Gradient Flow

1. Set error threshold.
2. Evaluate if $A$ is in range of $A_{opt}$ (which I think is proxied by $I$).
3. Evaluate if it is within time threshold.

- Run this test with existing settings, only adding the tracking above.
