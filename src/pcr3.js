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
    // this bucle find the principal components that has x percent value
    const sum = evalues.reduce((a, b) => a + b, 0);
    const weigth = evalues.map((x, i) => ({ weigth: (x / sum) * 100, evalues: evalues[i], componentNumber: i + 1 }));
    weigth.sort((a, b) => a.weigth < b.weigth);
    let n = 0; let z = 0; let l = 0;
    while (z < pcaWeight) {
      l = weigth[n].weigth; n++; z = z + l;
    }
    let predictorsMatrix = new Matrix(predictor);
    // this bucle find the liner regression coefficients in therms of principal components choosed
    const loadings = new Matrix(pca.getLoadings().data.slice(0, n));
    this.loadingsData = loadings.map((x, i) => ({ weigth: (evalues[i] / sum) * 100, evalues: evalues[i], componentNumber: i + 1, component: pca.getLoadings().data.slice(i, i + 1) }));
    
    
    
    let scores = predictorsMatrix.mmul(loadings.transpose());;
    this.scores = scores;
    let responseMatrix = new Matrix(response);
    const scoresLr = new MLR(scores, responseMatrix, { intercept: this.intercept });
    const coefficientsMatrix = new Matrix(scoresLr.toJSON().weights);
    console.log(coefficientsMatrix);
    console.log(loadings); // columns are fixed, the rows are the components, so if depends of the number of components choosed

  

    console.log(coefficientsMatrix);
    console.log(loadings);




    var ghj = coefficientsMatrix.mmul(loadings);


    


    this.ghj = ghj.transpose();


    // // Getting the regression coefficients in terms of initial data
    // let coefficients = [];
    // for (let i = 0; i < coefficientsMatrix.length; i++) {
    //   for (let j = 0; j < loadings.length; j++) {
    //     loadings.mulRow(j, coefficientsMatrix[i][j]);
    //   }
    //   let c = [coefficientsMatrix[i][coefficientsMatrix[0].length - 1]];
    //   for (let k = 0; k < loadings.transpose().length; k++) {
    //     let total = loadings.transpose()[k].reduce(function (a, b) {
    //       return a + b;
    //     });
    //     c.push(total);
    //   }
    //   coefficients.push(c);
    // }


    // let coefficientsData = new Matrix(coefficients);
    // this.coefficients = coefficientsData.transpose();
    // console.log(coefficientsData);
    // if (!this.intercept) {
    //   predictorsMatrix.addColumn(0, new Array(predictor.length).fill(1));
    // }
    console.log(predictorsMatrix.transpose());
    console.log(ghj);


    let prediction = ghj.mmul(predictorsMatrix.transpose());
    console.log(prediction);


    this.prediction = prediction;
    let residual = [];
    for (let j = 0; j < response.length; j++) {
      let g = [];
      for (let k = 0; k < response[0].length; k++) {
        g.push(response[j][k] - prediction[j][k]);
      }
      residual.push(g);
      g = [];
    } this.residual = residual;

    let xMedia = [];
    xMedia = predictor
      .map((a) => a.reduce((a, b) => (a + b), 0) / predictor[0].length);
    this.xMedia = xMedia;

    let yMedia = [];
    yMedia = response
      .map((a) => a.reduce((a, b) => (a + b), 0) / response[0].length);
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
      ssr.push(prediction[i].map((x) => Math.pow(x - yMedia[i], 2))
        .reduce((a, b) => a + b));
      yVariance.push(ssr[i] / (response[0].length - 1));
      stdDeviationY.push(Math.sqrt(yVariance[i]));
    }
    this.ssr = ssr;
    this.stdDeviationY = stdDeviationY;
    this.yVariance = yVariance;

    let sse = [];
    sse = residual.map((a) => a.map((x) => Math.pow(x, 2))
      .reduce((a, b) => a + b));
    this.sse = sse;

    let r2 = [];
    for (let i = 0; i < ssr.length; i++) {
      r2.push(ssr[i] / sst[i]);
    }
    this.r2 = r2;

    let xVariance = [];
    let stdDeviationX = [];
    for (let i = 0; i < predictor.length; i++) {
      xVariance.push(predictor[i].map((x) => Math.pow(x - xMedia[i], 2))
        .reduce((a, b) => a + b) / (predictor[0].length - 1));
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
  * @returns {[Matrix]}
  */
  predict(x) {
    const result = [];
    var g = [];
    if (this.intercept) {
      x.unshift(1);
    }
    for (let i = 0; i < this.ghj[0].length; i++) {
      for (let j = 0; j < this.ghj.length; j++) {
        g.push(this.ghj[j][i] * x[j]);
      }
      result[i] = g.reduce((a, b) => a + b);
      g = [];
    }
    //return this.coefficients;
    return result;
  }

  /**
  * Returns the regression coefficients
  * @returns {[Array]}
  */
  getStatistic() {
    return this.stat;
  }

  /**
  * Returns the regression coefficients
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
    return this.coefficients;
  }

  /**
  * Returns the number of principal components used
  * @returns {[number]}
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
