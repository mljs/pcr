'use strict';

const matrixLib = require('ml-matrix');
const PCA = require('ml-pca').PCA;
const MLR = require('ml-regression-multivariate-linear');

const Matrix = matrixLib.Matrix;

/**
 * Creates new PCR (Principal component regression)
 * @param {Array} predictor - matrix with predictor variables. (Each column is an array)
 * @param {Array} response - matrix with response variables. (Each column is an array)
 * @param {number} pcaWeight - Weight to choose the principal components. It refers to the weight that components must sum with each other (in percent) to perform the regression.
 * @param {boolean} intercept - Intercept
 * */

class PCR {
  constructor(predictor, response, options = {}) {
    const {
      intercept = true,
      pcaWeight = 1,
      nComp = undefined,
      scale = false,
      center = true,
    } = options;

    this.intercept = intercept;
    this.pcaWeight = pcaWeight;
    let pca;

    if (nComp) {
      pca = new PCA(predictor, {
        scale: scale,
        center: center,
        method: 'NIPALS',
        nCompNIPALS: nComp,
      });
    } else {
      pca = new PCA(predictor, { scale: scale, center: center });
    }

    let evalues = pca.getEigenvalues();

    const sum = evalues.reduce(
      (accumulator, currentValue) => accumulator + currentValue,
    );

    const weight = evalues.map((value, index) => ({
      weight: value / sum,
      evalues: evalues[index],
      componentNumber: index + 1,
    }));

    weight.sort((first, second) => first.weight < second.weight);
    let n = 0;
    let z = 0;
    let l = 0;

    while (z < this.pcaWeight) {
      l = weight[n].weight;
      z += l;
      n++;
    }

    let predictorsMatrix = new Matrix(predictor);
    let responseMatrix = new Matrix(response);

    const loadings = pca.getLoadings();
    const selectedLoadings = loadings.subMatrixRow(
      new Array(n).fill().map((value, index) => index),
    );
    this.loadingsData = new Array(selectedLoadings.rows)
      .fill()
      .map((value, index) => ({
        weight: (evalues[index] / sum) * 100,
        evalues: evalues[index],
        componentNumber: index + 1,
        component: loadings.getRow(index),
      }));

    let scores = predictorsMatrix.mmul(loadings.transpose());
    this.scores = scores;

    const scoresLr = new MLR(scores, responseMatrix, {
      intercept: this.intercept,
    });

    const coefficientsMatrix = new Matrix(
      scoresLr.toJSON().weights,
    ).transpose();
    const coefficients = coefficientsMatrix
      .subMatrixColumn(
        new Array(coefficientsMatrix.columns - 1)
          .fill()
          .map((value, index) => index),
      )
      .mmul(loadings.transpose())
      .transpose();

    if (this.intercept === true) {
      coefficients.addRow(
        0,
        coefficientsMatrix.getColumn(coefficientsMatrix.rows - 1),
      );
    }

    this.coefficients = coefficients;

    if (this.intercept === true) {
      predictorsMatrix.addColumn(0, new Array(predictor.length).fill(1));
    }

    let yFittedValues = predictorsMatrix.mmul(coefficients);
    this.yFittedValues = yFittedValues;

    let residual = responseMatrix.sub(yFittedValues).to2DArray();
    this.residual = residual;

    let xMedia = [];
    xMedia = predictor.map(
      (a) => a.reduce((a, b) => a + b, 0) / predictor[0].length,
    );
    this.xMedia = xMedia;

    let yMedia = [];
    yMedia = response.map(
      (a) => a.reduce((a, b) => a + b, 0) / response[0].length,
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
        yFittedValues.data[i]
          .map((x) => Math.pow(x - yMedia[i], 2))
          .reduce((a, b) => a + b),
      );
      yVariance.push(ssr[i] / (response[0].length - 1));
      stdDeviationY.push(Math.sqrt(yVariance[i]));
    }
    this.ssr = ssr;
    this.stdDeviationY = stdDeviationY;
    this.yVariance = yVariance;

    let sse = [];
    sse = residual.map((a) =>
      a.map((x) => Math.pow(x, 2)).reduce((a, b) => a + b),
    );
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
          (predictor[0].length - 1),
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
      stdDeviationX: this.stdDeviationX,
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
    for (let i = 0; i < this.coefficients.columns; i++) {
      for (let j = 0; j < this.coefficients.rows; j++) {
        g.push(this.coefficients.get(j, i) * x[j]);
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
