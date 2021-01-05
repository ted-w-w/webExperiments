const fs = require('fs');
const { createCanvas } = require('canvas')

const tf = require('@tensorflow/tfjs-node')

const trainingImageCount = 60000;
const k = 10;
const testImageCount = 10000;

// info on file structure here: http://yann.lecun.com/exdb/mnist/

const imageWidth = 28;
const imageHeight = 28;


const getLabelAtIndex = (index, trainingOrTest) => {

  // open file
  let fd = null;

  // open file
  if (trainingOrTest == "training") {
    fd = fs.openSync('./public/data/train-labels-idx1-ubyte', 'r');
  }
  else if (trainingOrTest == "test") {
    fd = fs.openSync('./public/data/t10k-labels-idx1-ubyte', 'r');
  }
  else {
    console.log("trainingOrTest argument not supplied correctly. should be supplied as either 'training' or 'test'. Using training as default");
    fd = fs.openSync('./public/data/train-labels-idx1-ubyte', 'r');
  }

  // create buffer for bytes
  let buffer = Buffer.alloc(4);

  // read number of labels (stored as 32 byte int at position 4)
  // fs.readSync(fd, buffer, 0, 4, 4);
  // let numberOfLabels = buffer.readUInt32BE();
  // console.log("Number of labels = ", numberOfLabels);

  // read label value at given index
  const positionOfFirstLabel = 8
  fs.readSync(fd, buffer, 0, 1, positionOfFirstLabel + index);
  const labelValue = buffer.readUInt8();

  // close the file
  fs.closeSync(fd);

  return labelValue;
}

const getPixelArrayForImageAtIndex = (index, trainingOrTest, width = imageWidth, height = imageHeight) => {

  let fd = null;

  // open file
  if (trainingOrTest == "training") {
    fd = fs.openSync('./public/data/train-images-idx3-ubyte', 'r');
  }
  else if (trainingOrTest == "test") {
    fd = fs.openSync('./public/data/t10k-images-idx3-ubyte', 'r');
  }
  else {
    console.log("trainingOrTest argument not supplied correctly. should be supplied as either 'training' or 'test'. Using training as default");
    fd = fs.openSync('./public/data/train-images-idx3-ubyte', 'r');
  }

  // create buffer for bytes

  const imageLength = width * height;

  let imageBuffer = Buffer.alloc(imageLength);

  // read label value at given index
  const positionOfFirstImage = 16;
  fs.readSync(fd, imageBuffer, 0, imageLength, positionOfFirstImage + (imageLength * index));

  let pixelArray = [];

  for (var i = 0; i < imageLength; i++) {
    const pixelValue = 255 - imageBuffer.readUInt8(i);
    pixelArray.push(pixelValue);
  }

  // close the file
  fs.closeSync(fd);

  return pixelArray;
}

const pixelArrayToPNG = (pixelArray, fileToWriteTo, width = imageWidth, height = imageHeight) => {

  const canvas = createCanvas(width, height)
  const context = canvas.getContext('2d')
  const imageLength = width * height;
  let imageData = context.getImageData(0, 0, width, height);

  for (var i = 0; i < imageLength; i++) {
    const pixelValue = pixelArray[i];
    imageData.data[i * 4] = pixelValue;
    imageData.data[i * 4 + 1] = pixelValue;
    imageData.data[i * 4 + 2] = pixelValue;
    imageData.data[i * 4 + 3] = 255;
  }

  context.putImageData(imageData, 0, 0);

  const buffer = canvas.toBuffer('image/png')
  fs.writeFileSync(fileToWriteTo, buffer)

}

const distanceBetween1DTensors = (tensorA, tensorB) => {
  return tensorA.sub(tensorB, 0).pow(2).sum(0).pow(0.5).arraySync();
}

let trainingImagesAsTensors = [];

console.log("importing training data...");

for (var i = 0; i < trainingImageCount; i++) {
  trainingImagesAsTensors.push(tf.tensor(getPixelArrayForImageAtIndex(i, "training")));
}

let numberCorrect = 0;

const predictFromImageTensor = (testTensor) => {

  const timeOfStart = new Date().getTime();

  let distances = [];

  // console.log("calculating distances...");

  for (var j = 0; j < trainingImageCount; j++) {
    const trainingTensor = trainingImagesAsTensors[j];
    const distance = distanceBetween1DTensors(testTensor, trainingTensor);
    distances.push({
      index: j,
      distance: distance
    });
  }

  const sortedDistances = distances.sort((a, b) => {
    return a.distance > b.distance ? 1 : -1;
  });

  const topKIndices = sortedDistances.slice(0, k);

  // console.log(topKIndices);

  let labelCounts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  for (let j = 0; j < topKIndices.length; j++) {
    const label = getLabelAtIndex(topKIndices[j].index, "training");
    labelCounts[label] += 1;
  }

  // console.log(labelCounts);

  let distancesSorted = labelCounts.map((count, index) => {
    return {
      index, count
    }
  });

  distancesSorted.sort((a, b) => {
    return a.count < b.count ? 1 : -1;
  });

  const prediction = distancesSorted[0].index;

  return prediction;

}


// run all tests

for (var i = 0; i < testImageCount; i++) {

  const timeOfStart = new Date().getTime();

  const testIndex = i;
  const testTensor = tf.tensor(getPixelArrayForImageAtIndex(testIndex, "test"));
  const testLabel = getLabelAtIndex(testIndex, "test");

  let distances = [];

  // console.log("calculating distances...");

  for (var j = 0; j < trainingImageCount; j++) {
    const trainingTensor = trainingImagesAsTensors[j];
    const distance = distanceBetween1DTensors(testTensor, trainingTensor);
    distances.push({
      index: j,
      distance: distance
    });
  }

  const sortedDistances = distances.sort((a, b) => {
    return a.distance > b.distance ? 1 : -1;
  });

  const topKIndices = sortedDistances.slice(0, k);

  // console.log(topKIndices);

  let labelCounts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  for (let j = 0; j < topKIndices.length; j++) {
    const label = getLabelAtIndex(topKIndices[j].index, "training");
    labelCounts[label] += 1;
  }

  // console.log(labelCounts);

  let distancesSorted = labelCounts.map((count, index) => {
    return {
      index, count
    }
  });

  distancesSorted.sort((a, b) => {
    return a.count < b.count ? 1 : -1;
  });

  const prediction = distancesSorted[0].index;

  // console.log("Truth:", testLabel, ", Prediction: ", prediction);

  if(prediction == testLabel){
    numberCorrect++;
  }

  const accuracy = numberCorrect/(i+1);


  const timeOfEnd = new Date().getTime();

  const timeSinceStart = (timeOfEnd - timeOfStart) / 1000;

  console.log("Training data: ", trainingImageCount)
  console.log("Tests: ", i+1)
  console.log("K:", k);
  console.log("Time taken:", timeSinceStart.toFixed(1), "s");
  console.log("Accuracy: ", Math.round(accuracy * 100), "%");
  console.log("---");
}
