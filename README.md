# PCR

Principal component regression.

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

## Installation

`$ npm install ml-pcr`

## Usage

```js
const { PCR } = require('pcr');

const x = [
  [0, 0],
  [1, 2],
  [2, 3],
  [3, 4],
];

const y = [
  [0, 0, 0],
  [2, 4, 3],
  [4, 6, 5],
  [6, 8, 7],
];

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

- Miller, J. N., & Miller, J. C. (2010). Statistics and Chemometrics for Analytical Chemistry.

- [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_regression).

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-pcr.svg
[npm-url]: https://npmjs.org/package/ml-pcr
[ci-image]: https://github.com/mljs/pcr/workflows/Node.js%20CI/badge.svg?branch=master
[ci-url]: https://github.com/mljs/pcr/actions?query=workflow%3A%22Node.js+CI%22
[codecov-image]: https://img.shields.io/codecov/c/github/mljs/ml-pcr.svg
[codecov-url]: https://codecov.io/gh/mljs/ml-pcr
[download-image]: https://img.shields.io/npm/dm/ml-pcr.svg
[download-url]: https://npmjs.org/package/ml-pcr
