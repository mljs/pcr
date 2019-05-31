'use strict';

const matrixLib = require('ml-matrix');
const MLR = require('ml-regression-multivariate-linear');
const PCA = require('ml-pca').PCA;

const Matrix = matrixLib.Matrix;

/**
 * Creates new MLR (Multiple Linear Regression)
 * @param {Array} predictor - matrix with predictor variables. (Each column is an array)
 * @param {Array} response - matrix with response variables. (Each column is an array)
 * @param {number} pcaWeight - percentage to choose the principal components
 * @param {boolean} intercept - Intercept
 * */

class PCR {
  constructor(predictor, response, intercept, pcaWeight) {
    this.intercept = intercept;
    if (!pcaWeight) {
      pcaWeight = 100;
    } else if (!intercept) {
      intercept = true;
    }
    let pca = new PCA(predictor);
    let evalues = pca.getEigenvalues();

    const sum = evalues.reduce((a, b) => a + b, 0);
    const weigth = evalues.map((x, i) => ({
      weigth: (x / sum) * 100,
      evalues: evalues[i],
      componentNumber: i + 1
    }));
    weigth.sort((a, b) => a.weigth < b.weigth);
    let n = 0;
    let z = 0;
    let l = 0;
    while (z < pcaWeight) {
      l = weigth[n].weigth;
      n++;
      z = z + l;
    }
    let predictorsMatrix = new Matrix(predictor);

    const loadings = new Matrix(pca.getLoadings().data.slice(0, n));
    this.loadingsData = loadings.map((x, i) => ({
      weigth: (evalues[i] / sum) * 100,
      evalues: evalues[i],
      componentNumber: i + 1,
      component: pca.getLoadings().data.slice(i, i + 1)
    }));
    let scores = predictorsMatrix.mmul(loadings.transpose());
    this.scores = scores;
    let responseMatrix = new Matrix(response);
    const scoresLr = new MLR(scores, responseMatrix, {
      intercept: this.intercept
    });
    const coefficientsMatrix = new Matrix(
      scoresLr.toJSON().weights
    ).transpose();
    let load = loadings.transpose();

    let g = [];
    let h = [];
    let coefficients = [];
    for (let k = 0; k < coefficientsMatrix.length; k++) {
      for (let i = 0; i < load.length; i++) {
        for (let j = 0; j < load[0].length; j++) {
          g.push(load[i][j] * coefficientsMatrix[k][j]);
        }
        h.push(g);
        g = [];
      }
      coefficients.push(new Matrix(h).map((a) => a.reduce((a, b) => a + b)));
      h = [];
    }
    if (this.intercept === true) {
      for (let i = 0; i < coefficients.length; i++) {
        coefficients[i].unshift(
          coefficientsMatrix[i][coefficientsMatrix[0].length - 1]
        );
      }
    }
    let coefficientsData = new Matrix(coefficients);
    this.coefficients = coefficientsData.transpose();

    if (this.intercept === true) {
      predictorsMatrix.addColumn(0, new Array(predictor.length).fill(1));
    }
    let yFittedValues = predictorsMatrix.mmul(coefficientsData.transpose());
    this.yFittedValues = yFittedValues;

    let residual = [];
    for (let j = 0; j < response.length; j++) {
      let g = [];
      for (let k = 0; k < response[0].length; k++) {
        g.push(response[j][k] - yFittedValues[j][k]);
      }
      residual.push(g);
      g = [];
    }
    this.residual = residual;

    let xMedia = [];
    xMedia = predictor.map(
      (a) => a.reduce((a, b) => a + b, 0) / predictor[0].length
    );
    this.xMedia = xMedia;

    let yMedia = [];
    yMedia = response.map(
      (a) => a.reduce((a, b) => a + b, 0) / response[0].length
    );
    this.yMedia = yMedia;

    let sst = [];
    for (let i = 0; i < response.length; i++) {
      sst[i] = response[i]
        .map((x) => Math.pow(x - yMedia[i], 2))
        .reduce((a, b) => a + b);
    }
    this.sst = sst;

    let ssr = [];
    let yVariance = [];
    let stdDeviationY = [];
    for (let i = 0; i < response.length; i++) {
      ssr.push(
        yFittedValues[i]
          .map((x) => Math.pow(x - yMedia[i], 2))
          .reduce((a, b) => a + b)
      );
      yVariance.push(ssr[i] / (response[0].length - 1));
      stdDeviationY.push(Math.sqrt(yVariance[i]));
    }
    this.ssr = ssr;
    this.stdDeviationY = stdDeviationY;
    this.yVariance = yVariance;

    let sse = [];
    sse = residual.map((a) => a.map((x) => Math.pow(x, 2)).reduce((a, b) => a + b));
    this.sse = sse;

    let r2 = [];
    for (let i = 0; i < ssr.length; i++) {
      r2.push(ssr[i] / sst[i]);
    }
    this.r2 = r2;

    let xVariance = [];
    let stdDeviationX = [];
    for (let i = 0; i < predictor.length; i++) {
      xVariance.push(
        predictor[i]
          .map((x) => Math.pow(x - xMedia[i], 2))
          .reduce((a, b) => a + b) /
          (predictor[0].length - 1)
      );
      stdDeviationX[i] = Math.sqrt(xVariance[i]);
    }
    this.xVariance = xVariance;
    this.stdDeviationX = stdDeviationX;

    let Statistic = {
      residuals: this.residual,
      yMedia: this.yMedia,
      xMedia: this.xMedia,
      SST: this.sst,
      SSR: this.ssr,
      SSE: this.sse,
      R2: this.r2,
      yVariance: this.yVariance,
      xVariance: this.xVariance,
      stdDeviationY: this.stdDeviationY,
      stdDeviationX: this.stdDeviationX
    };
    this.stat = Statistic;
  }

  /**
   * Predict y-values for a given x
   * @returns {[Array]}
   */
  predict(x) {
    const result = [];
    let g = [];
    if (this.intercept) {
      x.unshift(1);
    }
    for (let i = 0; i < this.coefficients[0].length; i++) {
      for (let j = 0; j < this.coefficients.length; j++) {
        g.push(this.coefficients[j][i] * x[j]);
      }
      result[i] = g.reduce((a, b) => a + b);
      g = [];
    }
    return result;
  }

  /**
   * Returns some basic statistics of the regression
   * @returns {[Array]}
   */
  getStatistic() {
    return this.stat;
  }

  /**
   * Returns fitted values of Y
   * @returns {[Array]}
   */
  getFittedValuesY() {
    return this.yFittedValues;
  }

  /**
   * Returns the regression coefficients
   * @returns {[Array]}
   */
  getCoefficients() {
    return this.coefficients;
  }

  /**
   * Returns the scores for principal components
   * @returns {[Object]}
   */
  getLoadingsdata() {
    return this.loadingsData;
  }

  /**
   * Returns the number of principal components used
   * @returns {[Array]}
   */
  getScores() {
    return this.scores;
  }
}
module.exports = PCR;
