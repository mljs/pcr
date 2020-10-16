# PCR

Principal component regression.

### new PCR(x, y, {options})

**Arguments**

* `x`: Matrix containing the inputs.
* `y`: Matrix containing the outputs.

**Options**

* `intercept`: boolean indicating if intercept terms should be computed.
* `weight`: number (0 - 1): It refers to the weight that components must sum with each otherto perform the regression. When this is exactly 1, pcr perform a multiple linear regression.

## Usage

```js

const PCR = require('pcr');

const x = [[0, 0], [1, 2], [2, 3], [3, 4]];

const y = [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]];

const pcr = new PCR(x, y, { intercept: true, weight: 1 });
console.log(pcr.predict([3, 3])); // Predict Y for an given X
// [6, 6, 6]

console.log(pcr.getLoadingsdata()); // Returns the information of loadings used to perform the linear regression
/*
{
    weigth: 99.20021500994476,
    evalues: 4.546676521289134,
    componentNumber: 1,
    component: [ [Array] ]
  },
  {
    weigth: 0.7997849900552465,
    evalues: 0.036656812044198794,
    componentNumber: 2,
    component: [ [Array] ]
  }
]
*/
```

## References
* Miller, J. N., & Miller, J. C. (2010). Statistics and Chemometrics for Analytical Chemistry.
* [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_regression).

## License
[MIT](./LICENSE)
