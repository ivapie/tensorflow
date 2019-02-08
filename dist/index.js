// modules are defined as an array
// [ module function, map of requires ]
//
// map of requires is short require name -> numeric require
//
// anything defined in a previous bundle is accessed via the
// orig method which is the require for previous bundles

// eslint-disable-next-line no-global-assign
parcelRequire = (function (modules, cache, entry, globalName) {
  // Save the require from previous bundle to this closure if any
  var previousRequire = typeof parcelRequire === 'function' && parcelRequire;
  var nodeRequire = typeof require === 'function' && require;

  function newRequire(name, jumped) {
    if (!cache[name]) {
      if (!modules[name]) {
        // if we cannot find the module within our internal map or
        // cache jump to the current global require ie. the last bundle
        // that was added to the page.
        var currentRequire = typeof parcelRequire === 'function' && parcelRequire;
        if (!jumped && currentRequire) {
          return currentRequire(name, true);
        }

        // If there are other bundles on this page the require from the
        // previous one is saved to 'previousRequire'. Repeat this as
        // many times as there are bundles until the module is found or
        // we exhaust the require chain.
        if (previousRequire) {
          return previousRequire(name, true);
        }

        // Try the node require function if it exists.
        if (nodeRequire && typeof name === 'string') {
          return nodeRequire(name);
        }

        var err = new Error('Cannot find module \'' + name + '\'');
        err.code = 'MODULE_NOT_FOUND';
        throw err;
      }

      localRequire.resolve = resolve;
      localRequire.cache = {};

      var module = cache[name] = new newRequire.Module(name);

      modules[name][0].call(module.exports, localRequire, module, module.exports, this);
    }

    return cache[name].exports;

    function localRequire(x){
      return newRequire(localRequire.resolve(x));
    }

    function resolve(x){
      return modules[name][1][x] || x;
    }
  }

  function Module(moduleName) {
    this.id = moduleName;
    this.bundle = newRequire;
    this.exports = {};
  }

  newRequire.isParcelRequire = true;
  newRequire.Module = Module;
  newRequire.modules = modules;
  newRequire.cache = cache;
  newRequire.parent = previousRequire;
  newRequire.register = function (id, exports) {
    modules[id] = [function (require, module) {
      module.exports = exports;
    }, {}];
  };

  for (var i = 0; i < entry.length; i++) {
    newRequire(entry[i]);
  }

  if (entry.length) {
    // Expose entry point to Node, AMD or browser globals
    // Based on https://github.com/ForbesLindesay/umd/blob/master/template.js
    var mainExports = newRequire(entry[entry.length - 1]);

    // CommonJS
    if (typeof exports === "object" && typeof module !== "undefined") {
      module.exports = mainExports;

    // RequireJS
    } else if (typeof define === "function" && define.amd) {
     define(function () {
       return mainExports;
     });

    // <script>
    } else if (globalName) {
      this[globalName] = mainExports;
    }
  }

  // Override the current require with this new one
  return newRequire;
})({"temperatures.json":[function(require,module,exports) {
module.exports = [{
  "temperatureC": 10,
  "value": "Baja"
}, {
  "temperatureC": 25,
  "value": "Baja"
}, {
  "temperatureC": 30,
  "value": "Baja"
}, {
  "temperatureC": 34,
  "value": "Baja"
}, {
  "temperatureC": 34.99,
  "value": "Baja"
}, {
  "temperatureC": 35,
  "value": "Normal"
}, {
  "temperatureC": 36,
  "value": "Normal"
}, {
  "temperatureC": 37,
  "value": "Normal"
}, {
  "temperatureC": 37.1,
  "value": "Alta"
}, {
  "temperatureC": 38,
  "value": "Alta"
}, {
  "temperatureC": 38.1,
  "value": "Muy Alta"
}, {
  "temperatureC": 40,
  "value": "Muy Alta"
}, {
  "temperatureC": 40.1,
  "value": "Extremadamente Alta"
}, {
  "temperatureC": 100,
  "value": "Extremadamente Alta"
}, {
  "temperatureC": 160,
  "value": "Extremadamente Alta"
}];
},{}],"index.js":[function(require,module,exports) {
"use strict";

var tf = _interopRequireWildcard(require("@tensorflow/tfjs"));

require("@tensorflow/tfjs-node");

var _temperatures = _interopRequireDefault(require("./temperatures.json"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = Object.defineProperty && Object.getOwnPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : {}; if (desc.get || desc.set) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } } newObj.default = obj; return newObj; } }

const trainingData = tf.tensor2d(_temperatures.default.map(item => [item.temperatureC]));
const outputData = tf.tensor2d(_temperatures.default.map(item => [item.value === "Baja" ? 1 : 0, item.value === "Normal" ? 1 : 0, item.value === "Alta" ? 1 : 0, item.value === "Muy Alta" ? 1 : 0, item.value === "Extremadamente Alta" ? 1 : 0]));
const testingData = tf.tensor2d(_temperatures.default.map(item => [item.temperatureC]));
const model = tf.sequential();
model.add(tf.layers.dense({
  inputShape: [1],
  activation: "sigmoid",
  units: 5
}));
model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 5
}));
model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.06)
}); // train/fit our network

const startTime = Date.now();
model.fit(trainingData, outputData, {
  epochs: 100
}).then(history => {
  // console.log(history)
  model.predict(testingData).print();
});
},{"./temperatures.json":"temperatures.json"}]},{},["index.js"], null)
//# sourceMappingURL=/index.map