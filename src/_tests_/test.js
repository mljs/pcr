'use strict';

// ···· Princiapl Component Regression ····//

var MLR = require("mlr");

const matrixLib = require('ml-matrix');

const Matrix = matrixLib.Matrix;

var PCR = require('pcr.js');

// *******************************************
// Example -->  Miller, J. N., & Miller, J. C. (2010). Statistics and Chemometrics for Analytical Chemistry. pp(238-243)
// dataset contains all the information for a multivariate problem, their three first columns corresponds to concentrations and the rest 6 columns corresponds to the absorbances for a UV experiment.
var data = new Matrix([
  [0.89, 0.02, 0.01, 18.7, 26.8, 42.1, 56.6, 70.0, 83.2],
  [0.46, 0.09, 0.24, 31.3, 33.4, 45.7, 49.3, 53.8, 55.3],
  [0.45, 0.16, 0.23, 30.0, 35.1, 48.3, 53.5, 59.2, 57.7],
  [0.56, 0.09, 0.09, 20.0, 25.7, 39.3, 46.6, 56.5, 57.8],
  [0.41, 0.02, 0.28, 31.5, 34.8, 46.5, 46.7, 48.5, 51.1],
  [0.44, 0.17, 0.14, 22.0, 28.0, 38.5, 46.7, 54.1, 53.6],
  [0.34, 0.23, 0.20, 25.7, 31.4, 41.1, 50.6, 53.5, 49.3],
  [0.74, 0.11, 0.01, 18.7, 26.8, 37.8, 50.6, 65.0, 72.3],
  [0.75, 0.01, 0.15, 27.3, 34.6, 47.8, 55.9, 67.9, 75.2],
  [0.48, 0.15, 0.06, 18.3, 22.8, 32.8, 43.4, 49.6, 51.1]
]);

var dataset = data.transpose();
var limit = 3; // the number of columb wich separate variables of responses.

// Building the matrix of predictor variables.
var predictor = []; //
for (var i = limit; i < dataset.length; i++) {
  predictor.push(dataset[i]);
}

// Building the response matrix.
var response = []; //
for (i = 0; i < limit; i++) {
  response.push(dataset[i]);
}
// *******************************************

const mlr = new MLR(predictor, response, true);

var pcr = new PCR(predictor, response, 99);

console.log('------> Coefficients of PCR');
console.log(pcr.getPCRcoeff());
console.log('------> Coefficients of MLR');
console.log(mlr.getCoefficients());
console.log('------> prediction using PCR');
console.log(pcr.getPCRprediction());
console.log('------> prediction using MLR');
console.log(mlr.getPrediction());
console.log('------>  Loadings data used to perform the linear regression');
console.log(pcr.getLoadingsdatalr());
console.log('------> Scores');
console.log(pcr.getScores());
console.log('------> Statistic information about the PCR performed');
console.log(pcr.getPCRstat());
