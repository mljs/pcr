'use strict';

var MLR = require("mlr");

const matrixLib = require('ml-matrix');

const PCA = require('ml-pca').PCA;

const Matrix = matrixLib.Matrix;

/**
 * Creates new MLR (Multiple Linear Regression)
 * @param {Array} predictor - matrix with predictor variables. (Each column is an array)
 * @param {Array} response - matrix with response variables. (Each column is an array)
 * @param {number} pcaweight - percentage to choose the principal components
 * */

class PCR {
  constructor(predictor, response, pcaweight) {
    if (!pcaweight) {
      pcaweight = 100;
    }

    var pca = new PCA(predictor[0].map((col, i) => predictor.map((row) => row[i])));

    var evalues = pca.getEigenvalues();
    // this bucle find the principal components that has x percent value
    const sum = evalues.reduce((a, b) => a + b, 0);
    const weigth = evalues.map((x, i) => ({ weigth: (x / sum) * 100, evalues: evalues[i], cnumber: i + 1 }));
    weigth.sort((a, b) => a.weigth < b.weigth);
    let n = 0; let z = 0; let l = 0;
    while (z < pcaweight) {
      l = weigth[n].weigth; n++; z = z + l;
    }

    // this bucle find the liner regression coefficients in therms of principal components choosed
    const loadings = new Matrix(pca.getLoadings().data.slice(0, n));
    this.loadingsdatalr = loadings.map((x, i) => ({ weigth: (evalues[i] / sum) * 100, evalues: evalues[i], cnumber: i + 1, ccomp: pca.getLoadings().data.slice(i, i+1) }));
    const scores = loadings.mmul(predictor);
    this.scores = scores;
    const scoreslr = new MLR(scores, response, true);
    const lrcoefficients = new Matrix(scoreslr.getCoefficients());

    // Getting the regression coefficients in terms of initial data
    var PCRcoeff = [];
    for (let i = 0; i < lrcoefficients.length; i++) {
      let load = new Matrix(JSON.parse(JSON.stringify(loadings)));
      for (let j = 1; j < lrcoefficients[0].length; j++) {
        load.mulRow(j - 1, lrcoefficients[i][j]);
      } let c = [lrcoefficients[i][0]];
      for (let k = 0; k < load.transpose().length; k++) {
        let total = load.transpose()[k].reduce(function (a, b) {
          return a + b;
        });
        c.push(total);
      }
      PCRcoeff.push(c);
    }
    this.PCRcoeff = PCRcoeff;

    var pcrcoeff = new Matrix(JSON.parse(JSON.stringify(this.PCRcoeff)));
    var onesvector = new Array(predictor[0].length).fill(1);
    var pred = new Matrix(JSON.parse(JSON.stringify(predictor)));
    pred.unshift(onesvector);
    var pcrprediction = pcrcoeff.mmul(pred);
    this.PCRprediction = pcrprediction;

    var residual = new Array(response.length);
    for (let j = 0; j < response.length; j++) {
      let g = new Array(response[0].length);
      for (let k = 0; k < response[0].length; k++) {
        g[k] = response[j][k] - pcrprediction[j][k];
      }
      residual[j] = g;
      g = [];
    } this.residual = residual;

    var xmedia = new Array(predictor.length);
    for (let i = 0; i < predictor.length; i++) {
      xmedia[i] = predictor[i].reduce((a, b) => (a + b), 0);
      xmedia[i] = xmedia[i] / predictor[0].length;
    } this.xmedia = xmedia;

    var ymedia = new Array(response.length);
    for (let i = 0; i < response.length; i++) {
      ymedia[i] = response[i].reduce((a, b) => (a + b), 0);
      ymedia[i] = ymedia[i] / response[0].length;
    } this.ymedia = ymedia;

    var sst = new Array(response.legth);
    for (let i = 0; i < response.length; i++) {
      sst[i] = response[i].map((x) => Math.pow(x - ymedia[i], 2));
      sst[i] = sst[i].reduce((a, b) => a + b);
    }
    this.sst = sst;

    var ssr = new Array(response.legth);
    var variancey = new Array(response.legth);
    var sdy = new Array(response.legth);
    for (let i = 0; i < response.length; i++) {
      ssr[i] = pcrprediction[i].map((x) => Math.pow(x - ymedia[i], 2));
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
      SST: this.sst,
      SSR: this.ssr,
      SSE: this.sse,
      R2: this.r2,
      VarianceY: this.variancey,
      StandarDeviationY: this.sdy
    };
    this.stat = Statistic;
  }
  /**
  * Returns the regression coefficients
  * @returns {[Array]}
  */
  getPCRstat() {
    return this.stat;
  }

  /**
  * Returns the regression coefficients
  * @returns {[Array]}
  */
  getPCRprediction() {
    return this.PCRprediction;
  }

  /**
  * Returns the regression coefficients
  * @returns {[Array]}
  */
  getPCRcoeff() {
    return this.PCRcoeff;
  }

  /**
  * Returns the number of principal components used
  * @returns {[number]}
  */
  getLoadingsdatalr() {
    return this.loadingsdatalr;
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
