# TODO

## Additional Changes

- I think we should test across different seeds, instead of just one.
- Investigate NaN gradients.
- Fix colour scheme in graphs.

## Different Distributions Test

Done.

## Non-Linear Distribution Test

Done.

## Conditional Number

- Verify implementation, see if conditional number is being set correctly.
- Compare performance with normal distribution.

## Softmax Attention - *Priority 2*

- Implement softmax attention
- Test against regular attention, compare convergence graphs.
- Do we still see the phenomen after looping above the loops it was trained on? Test for this.
- Display parameter graphs.

## Unconstraining P & Q

- Look into unconstraining the bias term, as well as other terms if possible.
- Implement and test against regular models across different loops.
- Test looping for $L_{test} > L_{train}$.

## Multiple Heads - *Priority 1*

- Test with multiple heads, and see how that will effect results.
- Compare against single head models for $1, 2, 5, 10$ layers.

## $L_{test} > L_{train}$ - *Priority 3*
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
