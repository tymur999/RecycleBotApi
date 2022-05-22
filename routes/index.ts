import {NextFunction, Request, Response} from "express";
import {PredictBody} from "../core/PredictBody";
import {GraphModel, Rank} from "@tensorflow/tfjs-node";
import {Tensor, Tensor3D} from "@tensorflow/tfjs";
import {TFSavedModel} from "@tensorflow/tfjs-node/dist/saved_model";
import {sleep} from "../utils";
const tf = require("@tensorflow/tfjs-node");
const express = require('express');
const router = express.Router();

const map = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'];

const optimalSize = 224;

let model: TFSavedModel;

tf.node.loadSavedModel("./public/model")
    .then(function(m: TFSavedModel) {
        model = m;
    });

/* POST home page. */
router.post('/', async function(req: Request, res: Response, _: NextFunction) {
    while(!model) {
        await sleep(100);
    }

    const body: PredictBody = req.body;
    if(!body || !body.image)
        return;

    const data = body.image.replace(
        /^data:image\/(png|jpeg);base64,/,
        ""
    );
    const buffer = Buffer.from(data, "base64");
    const tensor: Tensor3D = tf.node.decodeImage(buffer, 3);
    const smallImg: Tensor3D = tf.image.resizeBilinear(tensor, [optimalSize, optimalSize]);
    const input = tf.reshape(smallImg, [1,optimalSize, optimalSize, 3]);

    const prediction: Tensor<Rank> = model.predict(input) as Tensor<Rank>;
    const result = await prediction.data();
    const dictResult: any = {};
    for(let i = 0; i < map.length; i++) {
        dictResult[map[i]] = result[i];
    }

    return res.send(JSON.stringify(dictResult));
});

module.exports = router;
