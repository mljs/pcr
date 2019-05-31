'use strict';

// import PCR from '..';

// import MLR from '..';

var MLR = require('ml-regression-multivariate-linear');

var PCR = require('../src/pcr');

// *******************************************
// Example -->  Miller, J. N., & Miller, J. C. (2010). Statistics and Chemometrics for Analytical Chemistry. pp(238-243)

var response = [
  [0.89, 0.02, 0.01],
  [0.46, 0.09, 0.24],
  [0.45, 0.16, 0.23],
  [0.56, 0.09, 0.09],
  [0.41, 0.02, 0.28],
  [0.44, 0.17, 0.14],
  [0.34, 0.23, 0.2],
  [0.74, 0.11, 0.01],
  [0.75, 0.01, 0.15],
  [0.48, 0.15, 0.06]
];

var predictor = [
  [18.7, 26.8, 42.1, 56.6, 70.0, 83.2],
  [31.3, 33.4, 45.7, 49.3, 53.8, 55.3],
  [30.0, 35.1, 48.3, 53.5, 59.2, 57.7],
  [20.0, 25.7, 39.3, 46.6, 56.5, 57.8],
  [31.5, 34.8, 46.5, 46.7, 48.5, 51.1],
  [22.0, 28.0, 38.5, 46.7, 54.1, 53.6],
  [25.7, 31.4, 41.1, 50.6, 53.5, 49.3],
  [18.7, 26.8, 37.8, 50.6, 65.0, 72.3],
  [27.3, 34.6, 47.8, 55.9, 67.9, 75.2],
  [18.3, 22.8, 32.8, 43.4, 49.6, 51.1]
];

// *******************************************

describe('principal component regression', () => {
  it('should work with 2 inputs and 3 outputs', () => {
    const mlr = new MLR(
      [[0, 0], [1, 2], [2, 3], [3, 4]],
      // y0 = 2 * x0, y1 = 2 * x1, y2 = x0 + x1
      [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]]
    );

    const pcr = new PCR(
      [[0, 0], [1, 2], [2, 3], [3, 4]],
      // y0 = 2 * x0, y1 = 2 * x1, y2 = x0 + x1
      [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]]
    );
    expect(mlr.predict([2, 3]).map(Math.round)).toStrictEqual([4, 6, 5]);
    expect(mlr.predict([4, 4]).map(Math.round)).toStrictEqual([8, 8, 8]);
    expect(pcr.predict([2, 3]).map(Math.round)).toStrictEqual([4, 6, 5]);
    expect(pcr.predict([4, 4]).map(Math.round)).toStrictEqual([8, 8, 8]);
  });

  it('should work with 2 inputs and 3 outputs - intercept is 0', () => {
    const mlr = new MLR(
      [[0, 0], [1, 2], [2, 3], [3, 4]],
      // y0 = 2 * x0, y1 = 2 * x1, y2 = x0 + x1
      [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]],
      { intercept: true }
    );
    const pcr = new PCR(
      [[0, 0], [1, 2], [2, 3], [3, 4]],
      // y0 = 2 * x0, y1 = 2 * x1, y2 = x0 + x1
      [[0, 0, 0], [2, 4, 3], [4, 6, 5], [6, 8, 7]],
      { intercept: true}
    );

    expect(mlr.predict([2, 3]).map(Math.round)).toStrictEqual([4, 6, 5]);
    expect(mlr.predict([4, 4]).map(Math.round)).toStrictEqual([8, 8, 8]);
    expect(pcr.predict([2, 3]).map(Math.round)).toStrictEqual([4, 6, 5]);
    expect(pcr.predict([4, 4]).map(Math.round)).toStrictEqual([8, 8, 8]);
  });

  it('should work with 2 inputs and 3 outputs - intercept is not 0', () => {
    const mlr = new MLR(
      [[0, 0], [1, 2], [2, 3], [3, 4]],
      // y0 = 2 * x0 -1, y1 = 2 * x1 + 2, y2 = x0 + x1 + 10
      [[-1, 2, 10], [1, 6, 13], [3, 8, 15], [5, 10, 17]],
      { intercept: true }
    );
    const pcr = new PCR(
      [[0, 0], [1, 2], [2, 3], [3, 4]],
      // y0 = 2 * x0 -1, y1 = 2 * x1 + 2, y2 = x0 + x1 + 10
      [[-1, 2, 10], [1, 6, 13], [3, 8, 15], [5, 10, 17]],
      { intercept: true}
    );
    expect(pcr.predict([2, 3]).map(Math.round)).toStrictEqual([3, 8, 15]);
    expect(pcr.predict([4, 4]).map(Math.round)).toStrictEqual([7, 10, 18]);
    expect(mlr.predict([2, 3]).map(Math.round)).toStrictEqual([3, 8, 15]);
    expect(mlr.predict([4, 4]).map(Math.round)).toStrictEqual([7, 10, 18]);
  });

  it('Loadings data', () => {
    // Last argument refers to the weight that components must sum with each other (in percent) to perform the regression
    const pcr = new PCR(predictor, response, { intercept: true, pcaWeight: 0.99 });

    const loadings = pcr.getLoadingsdata();

    expect(loadings).toHaveLength(3);
    // First component eigenvalue and the weigth in the regression.
    expect(Math.round(loadings[0].evalues)).toStrictEqual(210);
    expect(Math.round(loadings[0].weight)).toStrictEqual(72);

    // Second component eigenvalue and the weigth in the regression.
    expect(Math.round(loadings[1].evalues)).toStrictEqual(74);
    expect(Math.round(loadings[1].weight)).toStrictEqual(25);

    // Third component eigenvalue and the weigth in the regression.
    expect(Math.round(loadings[2].evalues)).toStrictEqual(5);
    expect(Math.round(loadings[2].weight)).toStrictEqual(2);
  });
});
