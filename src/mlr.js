'use strict';

// ···· Univariate Multiple Linear Regression ····//

const matrixLib = require('ml-matrix');

const Matrix = matrixLib.Matrix;
const SVD = matrixLib.SVD;

/**
 * Creates new MLR (Univariate Multiple Linear Regression)
 * @param {Array} predictor - matrix with predictor variables. (Each column is an array)
 * @param {Array} response - matrix with response variables. (Each column is an array)
 *  @param {Boolean} intercept - matrix with response variables. (Each column is an array)
 * */

class MLR {
  constructor(predictor, response, intercept) {
    if (predictor) {
      var x = JSON.parse(JSON.stringify(predictor));
      if (intercept) {
        var c = new Array(predictor[0].length).fill(1);
        x.unshift(c);
      }
      var X = new Matrix(x);
      var Y = new Matrix(response);
      var xx = X.mmul(X.transpose());
      var xy = Y.mmul(X.transpose());
      var svdxx = new SVD(xx);
      var b = xy.mmul(svdxx.inverse());
      this.Coefficients = b;
    }
    var prediction = b.mmul(x);
    this.prediction = prediction;

    var residual = new Array(response.length);
    for (let i = 0; i < response.length; i++) {
      let nresidual = new Array(response[0].length);
      for (let j = 0; j < response[0].length; j++) {
        nresidual[j] = response[i][j] - prediction[i][j];
      }
      residual[i] = nresidual;
      nresidual = [];
    } this.residual = residual;

    var ymedia = new Array(response.length);
    for (let i = 0; i < response.length; i++) {
      ymedia[i] = response[i].reduce((a, b) => (a + b), 0);
      ymedia[i] = ymedia[i] / response[0].length;
    } this.ymedia = ymedia;

    var xmedia = new Array(predictor.length);
    for (let i = 0; i < predictor.length; i++) {
      xmedia[i] = predictor[i].reduce((a, b) => (a + b), 0);
      xmedia[i] = xmedia[i] / predictor[0].length;
    } this.xmedia = xmedia;

    var sst = new Array(response.legth);
    for (let i = 0; i < response.length; i++) {
      sst[i] = response[i].map((x) => Math.pow(x - ymedia[i], 2));
      sst[i] = sst[i].reduce((a, b) => (a + b), 0);
    }
    this.sst = sst;

    var ssr = new Array(response.legth);
    var variancey = new Array(response.legth);
    var sdy = new Array(response.legth);
    for (let i = 0; i < response.length; i++) {
      ssr[i] = prediction[i].map((x) => Math.pow(x - ymedia[i], 2));
      ssr[i] = ssr[i].reduce((a, b) => a + b);
      variancey[i] = ssr[i] / (response[0].length - 1);
      sdy[i] = Math.sqrt(variancey[i]);
    }
    this.ssr = ssr;
    this.sdy = sdy;
    this.variancey = variancey;

    var sse = new Array(response.legth);
    for (let i = 0; i < residual.length; i++) {
      sse[i] = residual[i].map((x) => Math.pow(x, 2));
      sse[i] = sse[i].reduce((a, b) => a + b);
    }
    this.sse = sse;

    var r2 = new Array(ssr.length);
    for (let i = 0; i < ssr.length; i++) {
      r2[i] = ssr[i] / sst[i];
    }
    this.r2 = r2;

    var variancex = new Array(predictor.length);
    var sdx = new Array(predictor.length);
    for (let i = 0; i < predictor.length; i++) {
      variancex[i] = predictor[i].map((x) => Math.pow(x - xmedia[i], 2));
      variancex[i] = variancex[i].reduce((a, b) => a + b);
      variancex[i] = variancex[i] / (predictor[0].length - 1);
      sdx[i] = Math.sqrt(variancex[i]);
    }
    this.variancex = variancex;
    this.sdx = sdx;

    var Statistic = {
      Residuals: this.residual,
      Ymedia: this.ymedia,
      Xmedia: this.xmedia,
      SST: this.sst,
      SSR: this.ssr,
      SSE: this.sse,
      R2: this.r2,
      VarianceY: this.variancey,
      VarianceX: this.variancex,
      StandarDeviationX: this.sdx,
      StandarDeviationY: this.sdy,
    };
    this.Statistic = Statistic;
  }
  /**
    * Returns the predicted responses
    * @returns {[Array]}
    */
  getPrediction() {
    return this.prediction;
  }
  /**
    * Returns the regression coefficients
    * @returns {[Array]}
    */
  getCoefficients() {
    return this.Coefficients;
  }

  /**
  * Returns the regression coefficients
  * @returns {[Array]}
  */
  getStatistic() {
    return this.Statistic;
  }
}

module.exports = MLR;
