'use strict';

var http = require('http');

var server = http.createServer();
function control(petic, resp) {
  resp.end();
}
server.on('request', control);
server.listen(8088);

// ···· Princiapl Component Regression ····//

var PCR = require('./pcr');
var MLR = require('ml-regression-multivariate-linear');

// *******************************************
// Example -->  Miller, J. N., & Miller, J. C. (2010). Statistics and Chemometrics for Analytical Chemistry. pp(238-243)
// dataset contains all the information for a multivariate problem, their three first columns corresponds to concentrations and the rest 6 columns corresponds to the absorbances for a UV experiment.

var response = [
  [0.89, 0.02, 0.01],
  [0.46, 0.09, 0.24],
  [0.45, 0.16, 0.23],
  [0.56, 0.09, 0.09],
  [0.41, 0.02, 0.28],
  [0.44, 0.17, 0.14],
  [0.34, 0.23, 0.20],
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
  [18.3, 22.8, 32.8, 43.4, 49.6, 51.1],
];

var x = [
  [0, 0],
  [1, 2],
  [2, 3],
  [3, 4]
];
var y = [
  [0, 0, 0],
  [2, 4, 3],
  [4, 6, 5],
  [6, 8, 7]
];
// *******************************************


// Predicting

//console.log(pcr.predict([1, 18.7, 26.8, 42.1, 56.6, 70.0, 83.2], [1, 31.3, 33.4, 45.7, 49.3, 53.8, 55.3])); // [0.895923203570272, 0.02473541079764563, 0.002473964766974257], [0.4610281652874223, 0.06631466894473925, 0.24264747076816687]
var pcr = new PCR(predictor, response, false, 99);
var mlr = new MLR(predictor, response, { intercept: false });

for (let i = 0; i < 9; i++) {
  console.log(mlr.predict(predictor[i]));
  console.log(pcr.predict(predictor[i]));
}

// console.log(pcr.getCoefficients());
// console.log(pcr.getPrediction());
console.log(pcr.getLoadingsdata()[0]);
console.log(Math.round(pcr.getLoadingsdata()[2].evalues)) //.map(Math.round));
console.log(Math.round(pcr.getLoadingsdata()[2].weigth)) //.map(Math.round));
// console.log(pcr.getScores());
// console.log(pcr.getStatistic());

//console.log(pcr.predict([18.7, 26.8, 42.1, 56.6, 70.0, 83.2])); // The data must be a matrix that has at least 2 rows

// const pcr2 = new PCR(x, y, true, 100);
// console.log(pcr2.getPrediction());
// const mlr2 = new MLR(x, y);
// console.log(mlr2.predict([2, 3]));
// console.log(pcr2.predict([2, 3]));

