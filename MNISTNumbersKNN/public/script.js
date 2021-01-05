const canvas = document.getElementById("canvas");

const width = 28
const height = 28

const displayWidth = 200;
const displayHeight = 200;

const canvasScale = width / displayWidth;

canvas.width = width;
canvas.height = height;

const context = canvas.getContext("2d");
context.fillStyle = "#fff"
context.fillRect(0, 0, width, height);

let mouseIsDown = false;

const brushSize = 10;

canvas.addEventListener('mousemove', (event) => {
  if (mouseIsDown) {
    // console.log("mouse moving in canvas");
    context.fillStyle = "#000"
    const topLeftX = (event.clientX - canvas.getBoundingClientRect().left) - (brushSize / 2);
    const topLeftY = (event.clientY - canvas.getBoundingClientRect().top) - (brushSize / 2);
    // console.log(topLeftX, topLeftY);
    context.fillRect(topLeftX * canvasScale, topLeftY * canvasScale, brushSize * canvasScale, brushSize * canvasScale);
  }
})

canvas.addEventListener('mousedown', () => {
  mouseIsDown = true;
})

canvas.addEventListener('mouseup', () => {
  mouseIsDown = false;
})

document.getElementById("buttonClear").onclick = () => {
  context.fillStyle = "#fff"
  context.fillRect(0, 0, width, height);
};


var testLabelsFileRequest = new XMLHttpRequest();
testLabelsFileRequest.open("GET", "./data/t10k-labels-idx1-ubyte", true);
testLabelsFileRequest.responseType = "arraybuffer";

var trainingLabelsFileRequest = new XMLHttpRequest();
trainingLabelsFileRequest.open("GET", "./data/train-labels-idx1-ubyte", true);
trainingLabelsFileRequest.responseType = "arraybuffer";

var testImagesFileRequest = new XMLHttpRequest();
testImagesFileRequest.open("GET", "./data/t10k-images-idx3-ubyte", true);
testImagesFileRequest.responseType = "arraybuffer";

var trainingImagesFileRequest = new XMLHttpRequest();
trainingImagesFileRequest.open("GET", "./data/train-images-idx3-ubyte", true);
trainingImagesFileRequest.responseType = "arraybuffer";

let testLabelsFileByteArray = null;
let trainingLabelsFileByteArray = null;
let testImagesFileByteArray = null;
let trainingImagesFileByteArray = null;

testLabelsFileRequest.onload = function () {
  var arrayBuffer = testLabelsFileRequest.response;
  console.log("Test Labels File loaded");
  testLabelsFileByteArray = new Uint8Array(arrayBuffer);
};


trainingLabelsFileRequest.onload = function () {
  var arrayBuffer = trainingLabelsFileRequest.response;
  console.log("Training Labels File loaded");
  trainingLabelsFileByteArray = new Uint8Array(arrayBuffer);
};


testImagesFileRequest.onload = function () {
  var arrayBuffer = testImagesFileRequest.response;
  console.log("Test Images File loaded");
  testImagesFileByteArray = new Uint8Array(arrayBuffer);
};


trainingImagesFileRequest.onload = function () {
  var arrayBuffer = trainingImagesFileRequest.response;
  console.log("Training Images File loaded");
  trainingImagesFileByteArray = new Uint8Array(arrayBuffer);
};

const waitUntilAllFilesLoaded = () => {
  if (testLabelsFileByteArray && testImagesFileByteArray && trainingLabelsFileByteArray && trainingImagesFileByteArray) {
    console.log("All files loaded");
    main();
  }
  else {
    console.log("Waiting for all files to load...");
    setTimeout(waitUntilAllFilesLoaded, 1000);
  }
}

console.log("sending requests");
testLabelsFileRequest.send(null);
trainingLabelsFileRequest.send(null);
testImagesFileRequest.send(null);
trainingImagesFileRequest.send(null);

waitUntilAllFilesLoaded();

const main = () => {

  const trainingImageCount = 600;
  const k = 10;

  const imageWidth = 28;
  const imageHeight = 28;

  const getLabelAtIndex = (index, trainingOrTest) => {

    let fileBytesArray = null;

    // open file
    if (trainingOrTest == "training") {
      fileBytesArray = trainingLabelsFileByteArray;
    }
    else if (trainingOrTest == "test") {
      fileBytesArray = testLabelsFileByteArray;
    }
    else {
      console.log("trainingOrTest argument not supplied correctly. should be supplied as either 'training' or 'test'. Using training as default");
      fileBytesArray = trainingLabelsFileByteArray;
    }

    // read label value at given index
    const positionOfFirstLabel = 8
    const labelValue = fileBytesArray[positionOfFirstLabel + index];

    return labelValue;
  }

  const getPixelArrayForImageAtIndex = (index, trainingOrTest, width = imageWidth, height = imageHeight) => {

    let fileBytesArray = null;

    // open file
    if (trainingOrTest == "training") {
      fileBytesArray = trainingImagesFileByteArray;
    }
    else if (trainingOrTest == "test") {
      fileBytesArray = testImagesFileByteArray;
    }
    else {
      console.log("trainingOrTest argument not supplied correctly. should be supplied as either 'training' or 'test'. Using training as default");
      fileBytesArray = trainingImagesFileByteArray;
    }

    const imageLength = width * height;

    // read label value at given index
    const positionOfFirstImage = 16;

    let pixelArray = [];

    for (var i = 0; i < imageLength; i++) {
      const pixelValue = 255 - fileBytesArray[positionOfFirstImage + (index * imageLength) + i];
      pixelArray.push(pixelValue);
    }

    return pixelArray;
  }

  const distanceBetween1DTensors = (tensorA, tensorB) => {
    return tensorA.sub(tensorB, 0).pow(2).sum(0).pow(0.5).arraySync();
  }

  let trainingImagesAsTensors = [];

  console.log("importing training data...");

  for (var i = 0; i < trainingImageCount; i++) {
    trainingImagesAsTensors.push(tf.tensor(getPixelArrayForImageAtIndex(i, "training")));
  }

  console.log("training data imported");

  document.getElementById("message").innerHTML = "Ready";

  const predictFromImageTensor = (testTensor) => {

    return new Promise((resolve, reject) => {

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

      resolve(prediction);

    });

  }

  const buttonPredict = document.getElementById("buttonPredict");

  buttonPredict.onclick = () => {

    document.getElementById("message").innerHTML = "Thinking...";
    console.log("thinking");

    let pixelArray = []
    const imageData = context.getImageData(0, 0, width, height);
    for (i = 0; i < imageData.data.length; i += 4) {
      pixelArray.push(imageData.data[i]);
    }

    setTimeout(() => {
      predictFromImageTensor(tf.tensor(pixelArray))
      .then((prediction) => {
        console.log("Prediction: ", prediction);

        document.getElementById("message").innerHTML = "I think you drew a " + prediction.toString();
      })
    }, 500);


  }
}