import { Matrix } from 'ml-matrix';
import { PCA } from 'ml-pca';
import MLR from 'ml-regression-multivariate-linear';

/**
 * Creates new PCR (Principal component regression)
 * @param {Array} predictor - matrix with predictor variables. (Each column is an array)
 * @param {Array} response - matrix with response variables. (Each column is an array)
 * @param {number} pcaWeight - Weight to choose the principal components. It refers to the weight that components must sum with each other (in percent) to perform the regression.
 * @param {boolean} intercept - Intercept
 * */

export class PCR {
  intercept: boolean;
  pcaWeight: number;
  loadingsData: {
    weight: number;
    evalues: number;
    componentNumber: number;
    component: number[];
  }[];
  scores: Matrix;
  coefficients: Matrix;
  yFittedValues: Matrix;
  xMedia?: number[];
  statistics: Statistics;

  constructor(
    predictor: number[][] | Matrix,
    response: number[][] | Matrix,
    options: Options = {},
  ) {
    const {
      intercept = true,
      pcaWeight = 1,
      nComp = 0,
      scale = false,
      center = true,
    } = options;

    this.intercept = intercept;
    this.pcaWeight = pcaWeight;
    let pca;

    if (nComp) {
      pca = new PCA(predictor, {
        scale,
        center,
        method: 'NIPALS',
        nCompNIPALS: nComp,
      });
    } else {
      pca = new PCA(predictor, { scale, center });
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

    weight.sort((first, second) => first.weight + second.weight);

    let n = 0;
    let z = 0;
    let l = 0;
    while (z < this.pcaWeight) {
      l = weight[n].weight;
      z += l;
      n++;
    }

    let predictorsMatrix = Matrix.checkMatrix(predictor);
    let responseMatrix = Matrix.checkMatrix(response);

    const loadings = pca.getLoadings();
    const selectedLoadings = loadings.subMatrixRow(
      new Array(n).fill(0).map((_, index) => index),
    );
    this.loadingsData = new Array(selectedLoadings.rows)
      .fill(0)
      .map((_, index) => ({
        weight: (evalues[index] / sum) * 100,
        evalues: evalues[index],
        componentNumber: index + 1,
        component: loadings.getRow(index),
      }));

    let scores = predictorsMatrix.mmul(loadings.transpose());
    this.scores = scores;

    const regressionScores = new MLR(scores, responseMatrix, {
      intercept: this.intercept,
    });

    const coefficientsMatrix = new Matrix(regressionScores.weights).transpose();
    const coefficients = coefficientsMatrix
      .subMatrixColumn(
        new Array(coefficientsMatrix.columns - 1)
          .fill(0)
          .map((_, index) => index),
      )
      .mmul(loadings.transpose())
      .transpose();

    if (this.intercept) {
      coefficients.addRow(
        0,
        coefficientsMatrix.getColumn(coefficientsMatrix.rows - 1),
      );
      predictorsMatrix.addColumn(0, new Array(predictorsMatrix.rows).fill(1));
    }

    this.coefficients = coefficients;

    let yFittedValues = predictorsMatrix.mmul(coefficients);
    this.yFittedValues = yFittedValues;

    let residual = responseMatrix.sub(yFittedValues).to2DArray();

    let xMedia = predictorsMatrix.mean('row');

    let yMedia = responseMatrix.mean('row');

    let sst = [];
    for (let i = 0; i < responseMatrix.rows; i++) {
      sst[i] = responseMatrix
        .getRow(i)
        .map((x) => Math.pow(x - yMedia[i], 2))
        .reduce((a, b) => a + b);
    }

    let ssr = [];
    let yVariance = [];
    let stdDeviationY = [];
    for (let i = 0; i < responseMatrix.rows; i++) {
      ssr.push(
        yFittedValues
          .getRow(i)
          .map((x) => Math.pow(x - yMedia[i], 2))
          .reduce((a, b) => a + b),
      );
      yVariance.push(ssr[i] / (responseMatrix.columns - 1));
      stdDeviationY.push(Math.sqrt(yVariance[i]));
    }

    let sse = [];
    sse = residual.map((a) =>
      a.map((x) => Math.pow(x, 2)).reduce((a, b) => a + b),
    );

    let r2 = [];
    for (let i = 0; i < ssr.length; i++) {
      r2.push(ssr[i] / sst[i]);
    }

    let xVariance = [];
    let stdDeviationX = [];
    for (let i = 0; i < predictorsMatrix.rows; i++) {
      xVariance.push(
        predictorsMatrix
          .getRow(i)
          .map((x) => Math.pow(x - xMedia[i], 2))
          .reduce((a, b) => a + b) /
          (predictorsMatrix.columns - 1),
      );
      stdDeviationX[i] = Math.sqrt(xVariance[i]);
    }

    let Statistic = {
      residual,
      yMedia,
      xMedia,
      SST: sst,
      SSR: ssr,
      SSE: sse,
      R2: r2,
      yVariance,
      xVariance,
      stdDeviationY,
      stdDeviationX,
    };
    this.statistics = Statistic;
  }

  /**
   * Predict y-values for a given x
   * @returns {[Array]}
   */
  predict(x: number[]) {
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
    return this.statistics;
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

export interface Options {
  intercept?: boolean;
  pcaWeight?: number;
  nComp?: number;
  scale?: boolean;
  center?: boolean;
}

export interface Statistics {
  residual: number[][];
  yMedia: number[];
  xMedia?: number[];
  SST: number[];
  SSR: number[];
  SSE: number[];
  R2: number[];
  yVariance: number[];
  xVariance: number[];
  stdDeviationY: number[];
  stdDeviationX: number[];
}
