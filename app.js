// ============================================================
// SonoRestore AI
// Browser-side ONNX ultrasound image restoration
// ============================================================

const MODEL_PATH = "./sonorestore_ai.onnx";

let session = null;
let selectedFile = null;


// ------------------------------------------------------------
// DOM ELEMENTS
// ------------------------------------------------------------

const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const restoreButton = document.getElementById("restoreButton");
const statusText = document.getElementById("status");

const resultsSection = document.getElementById("resultsSection");

const originalImage = document.getElementById("originalImage");
const bicubicImage = document.getElementById("bicubicImage");
const restoredImage = document.getElementById("restoredImage");

const downloadButton = document.getElementById("downloadButton");


// ------------------------------------------------------------
// LOAD ONNX MODEL
// ------------------------------------------------------------

async function loadModel() {

    try {

        statusText.textContent = "Loading SonoRestore AI model...";

        session = await ort.InferenceSession.create(
            MODEL_PATH,
            {
                executionProviders: ["wasm"]
            }
        );

        console.log("Model loaded successfully");

        console.log(
            "Inputs:",
            session.inputNames
        );

        console.log(
            "Outputs:",
            session.outputNames
        );

        statusText.textContent =
            "✅ AI model ready. Upload an ultrasound image.";

    } catch (error) {

        console.error(error);

        statusText.textContent =
            "❌ Could not load AI model. Make sure sonorestore_ai.onnx is in the same folder.";

    }
}


// ------------------------------------------------------------
// IMAGE FILE SELECTION
// ------------------------------------------------------------

imageInput.addEventListener("change", function () {

    if (!this.files || this.files.length === 0) {
        return;
    }

    selectedFile = this.files[0];

    fileName.textContent = selectedFile.name;

    restoreButton.disabled = false;

    statusText.textContent =
        "Image selected. Click Restore Image.";

    // Show original preview
    const url = URL.createObjectURL(selectedFile);

    originalImage.src = url;

});


// ------------------------------------------------------------
// LOAD IMAGE INTO HTML IMAGE
// ------------------------------------------------------------

function loadImage(file) {

    return new Promise((resolve, reject) => {

        const img = new Image();

        img.onload = () => resolve(img);

        img.onerror = () =>
            reject(new Error("Could not read image."));

        img.src = URL.createObjectURL(file);

    });

}


// ------------------------------------------------------------
// CREATE CANVAS
// ------------------------------------------------------------

function createCanvas(width, height) {

    const canvas = document.createElement("canvas");

    canvas.width = width;
    canvas.height = height;

    return canvas;

}


// ------------------------------------------------------------
// CONVERT IMAGE TO GRAYSCALE
// ------------------------------------------------------------

function convertToGrayscale(canvas) {

    const ctx = canvas.getContext("2d");

    const imageData =
        ctx.getImageData(
            0,
            0,
            canvas.width,
            canvas.height
        );

    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {

        const gray =
            0.299 * data[i] +
            0.587 * data[i + 1] +
            0.114 * data[i + 2];

        data[i] = gray;
        data[i + 1] = gray;
        data[i + 2] = gray;

    }

    ctx.putImageData(imageData, 0, 0);

}


// ------------------------------------------------------------
// CREATE 256×256 GRAYSCALE IMAGE
// ------------------------------------------------------------

function prepareOriginalImage(img) {

    const canvas = createCanvas(256, 256);

    const ctx = canvas.getContext("2d");

    ctx.drawImage(
        img,
        0,
        0,
        256,
        256
    );

    convertToGrayscale(canvas);

    return canvas;

}


// ------------------------------------------------------------
// DOWNSCALE TO 128×128
// ------------------------------------------------------------

function downscaleImage(canvas256) {

    const canvas128 =
        createCanvas(128, 128);

    const ctx =
        canvas128.getContext("2d");

    ctx.imageSmoothingEnabled = true;

    ctx.imageSmoothingQuality = "high";

    ctx.drawImage(
        canvas256,
        0,
        0,
        128,
        128
    );

    return canvas128;

}


// ------------------------------------------------------------
// ADD SIMULATED SPECKLE-LIKE NOISE
// ------------------------------------------------------------

function addSpeckleNoise(canvas) {

    const ctx = canvas.getContext("2d");

    const imageData =
        ctx.getImageData(
            0,
            0,
            canvas.width,
            canvas.height
        );

    const data = imageData.data;

    /*
     * Multiplicative noise approximation.
     * This creates a speckle-like degradation.
     */

    for (let i = 0; i < data.length; i += 4) {

        const original =
            data[i] / 255.0;

        // Random multiplicative variation
        const noise =
            0.75 + Math.random() * 0.5;

        let value =
            original * noise;

        value =
            Math.max(
                0,
                Math.min(1, value)
            );

        value *= 255;

        data[i] = value;
        data[i + 1] = value;
        data[i + 2] = value;

    }

    ctx.putImageData(
        imageData,
        0,
        0
    );

    return canvas;

}


// ------------------------------------------------------------
// CREATE INPUT TENSOR
// ------------------------------------------------------------

function canvasToTensor(canvas) {

    const ctx =
        canvas.getContext("2d");

    const imageData =
        ctx.getImageData(
            0,
            0,
            canvas.width,
            canvas.height
        );

    const data =
        imageData.data;

    const floatData =
        new Float32Array(
            128 * 128
        );

    for (
        let i = 0;
        i < 128 * 128;
        i++
    ) {

        floatData[i] =
            data[i * 4] / 255.0;

    }

    return new ort.Tensor(
        "float32",
        floatData,
        [1, 1, 128, 128]
    );

}


// ------------------------------------------------------------
// BICUBIC BASELINE
// ------------------------------------------------------------

function createBicubicImage(canvas128) {

    const canvas256 =
        createCanvas(256, 256);

    const ctx =
        canvas256.getContext("2d");

    ctx.imageSmoothingEnabled = true;

    ctx.imageSmoothingQuality = "high";

    ctx.drawImage(
        canvas128,
        0,
        0,
        256,
        256
    );

    return canvas256;

}


// ------------------------------------------------------------
// AI INFERENCE
// ------------------------------------------------------------

async function runAI(canvas128) {

    const inputTensor =
        canvasToTensor(canvas128);

    const inputName =
        session.inputNames[0];

    const outputName =
        session.outputNames[0];

    const feeds = {};

    feeds[inputName] =
        inputTensor;

    const results =
        await session.run(feeds);

    const output =
        results[outputName];

    return output;

}


// ------------------------------------------------------------
// CONVERT MODEL OUTPUT TO CANVAS
// ------------------------------------------------------------

function tensorToCanvas(outputTensor) {

    const canvas =
        createCanvas(256, 256);

    const ctx =
        canvas.getContext("2d");

    const imageData =
        ctx.createImageData(
            256,
            256
        );

    const output =
        outputTensor.data;

    for (
        let i = 0;
        i < 256 * 256;
        i++
    ) {

        let value =
            output[i];

        value =
            Math.max(
                0,
                Math.min(1, value)
            );

        const pixel =
            Math.round(
                value * 255
            );

        imageData.data[i * 4] =
            pixel;

        imageData.data[i * 4 + 1] =
            pixel;

        imageData.data[i * 4 + 2] =
            pixel;

        imageData.data[i * 4 + 3] =
            255;

    }

    ctx.putImageData(
        imageData,
        0,
        0
    );

    return canvas;

}


// ------------------------------------------------------------
// CANVAS → DATA URL
// ------------------------------------------------------------

function canvasToDataURL(canvas) {

    return canvas.toDataURL(
        "image/png"
    );

}


// ------------------------------------------------------------
// RESTORE BUTTON
// ------------------------------------------------------------

restoreButton.addEventListener(
    "click",
    async function () {

        if (!selectedFile) {

            alert(
                "Please select an ultrasound image first."
            );

            return;

        }

        if (!session) {

            alert(
                "AI model is still loading."
            );

            return;

        }

        try {

            restoreButton.disabled = true;

            statusText.textContent =
                "Processing image with SonoRestore AI...";

            resultsSection.classList.remove(
                "hidden"
            );


            // --------------------------------------------
            // LOAD IMAGE
            // --------------------------------------------

            const img =
                await loadImage(
                    selectedFile
                );


            // --------------------------------------------
            // PREPARE 256×256 IMAGE
            // --------------------------------------------

            const originalCanvas =
                prepareOriginalImage(
                    img
                );


            // --------------------------------------------
            // CREATE 128×128 INPUT
            // --------------------------------------------

            const lowResCanvas =
                downscaleImage(
                    originalCanvas
                );


            // --------------------------------------------
            // SIMULATE DEGRADATION
            // --------------------------------------------

            addSpeckleNoise(
                lowResCanvas
            );


            // --------------------------------------------
            // DISPLAY DEGRADED IMAGE
            // --------------------------------------------

            originalImage.src =
                canvasToDataURL(
                    lowResCanvas
                );


            // --------------------------------------------
            // BICUBIC
            // --------------------------------------------

            const bicubicCanvas =
                createBicubicImage(
                    lowResCanvas
                );

            bicubicImage.src =
                canvasToDataURL(
                    bicubicCanvas
                );


            // --------------------------------------------
            // AI RESTORATION
            // --------------------------------------------

            const output =
                await runAI(
                    lowResCanvas
                );


            console.log(
                "AI output:",
                output
            );


            // --------------------------------------------
            // CONVERT OUTPUT
            // --------------------------------------------

            const restoredCanvas =
                tensorToCanvas(
                    output
                );


            restoredImage.src =
                canvasToDataURL(
                    restoredCanvas
                );


            // Save result for download
            downloadButton.onclick =
                function () {

                    const link =
                        document.createElement("a");

                    link.download =
                        "sonorestore_restored.png";

                    link.href =
                        canvasToDataURL(
                            restoredCanvas
                        );

                    link.click();

                };


            statusText.textContent =
                "✅ Restoration complete!";

            restoreButton.disabled =
                false;

        } catch (error) {

            console.error(
                "Restoration error:",
                error
            );

            statusText.textContent =
                "❌ Error during restoration. Check the browser console.";

            restoreButton.disabled =
                false;

        }

    }
);


// ------------------------------------------------------------
// START
// ------------------------------------------------------------

loadModel();