import {NextFunction, Request, Response} from "express";
import {PredictBody} from "../core/PredictBody";
import {GraphModel} from "@tensorflow/tfjs-node";
import {Tensor3D} from "@tensorflow/tfjs";
const tf = require("@tensorflow/tfjs-node");
const express = require('express');
const router = express.Router();


const modelJson = "https://f000.backblazeb2.com/file/Kiwoon-Learning/tfjs/model.json";
let model: GraphModel;

tf.loadGraphModel(modelJson)
    .then(function (mdl: GraphModel) {
        model = mdl;
    });

/* POSt home page. */
router.post('/', async function(req: Request, res: Response, _: NextFunction) {
    while(!model) {
        await sleep(100);
    }

    const body: PredictBody = req.body;
    if(!body || !body.image)
        return;

    const buffer = Buffer.from(body.image, "base64");
    const tensor: Tensor3D = tf.browser.fromPixels(buffer);
    console.log(model.predict(tensor));

    return res.send(200);
});

function sleep(ms: number) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

module.exports = router;
