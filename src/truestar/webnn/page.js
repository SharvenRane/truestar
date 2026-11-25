// Truestar WebNN page: builds one MLGraph from a description made by truestar/webnn/translate.py
// and runs it. Every description node is one MLGraphBuilder call; an op the builder rejects is an
// error, never a fallback to another runtime.
(() => {
  const TYPED = {float32: Float32Array, int32: Int32Array, int64: BigInt64Array, uint8: Uint8Array, int8: Int8Array};

  function bytes(b64) {
    const s = atob(b64);
    const u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u.buffer;
  }

  function b64(buf) {
    const u = new Uint8Array(buf);
    let s = "";
    for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
    return btoa(s);
  }

  function opt(o, keys) {
    const r = {};
    for (const k of keys) if (o[k] !== undefined && o[k] !== null) r[k] = o[k];
    return r;
  }

  function emit(b, env, n) {
    const x = (i) => {
      const v = env[n.inputs[i]];
      if (v === undefined) throw new Error(`node ${n.name}: input ${n.inputs[i]} is not defined`);
      return v;
    };
    const ref = (name) => (name ? env[name] : undefined);
    switch (n.op) {
      case "add": case "sub": case "mul": case "div": case "pow": case "max": case "min": case "matmul":
        return b[n.op](x(0), x(1));
      case "relu": case "sigmoid": case "tanh": case "erf": case "sqrt": case "exp": case "log":
      case "abs": case "neg": case "reciprocal": case "identity": case "hardSwish": case "floor":
      case "ceil": case "sin": case "cos": case "softsign": case "softplus":
        return b[n.op](x(0));
      case "reshape": return b.reshape(x(0), n.shape);
      case "gemm": {
        const o = opt(n, ["alpha", "beta", "aTranspose", "bTranspose"]);
        if (n.c) o.c = ref(n.c);
        return b.gemm(x(0), x(1), o);
      }
      case "softmax": return b.softmax(x(0), n.axis);
      case "layerNormalization": {
        const o = opt(n, ["axes", "epsilon"]);
        if (n.scale) o.scale = ref(n.scale);
        if (n.bias) o.bias = ref(n.bias);
        return b.layerNormalization(x(0), o);
      }
      case "reduceMean": return b.reduceMean(x(0), opt(n, ["axes", "keepDimensions"]));
      case "transpose": return b.transpose(x(0), {permutation: n.permutation});
      case "conv2d": {
        const o = opt(n, ["padding", "strides", "dilations", "groups"]);
        if (n.bias) o.bias = ref(n.bias);
        return b.conv2d(x(0), x(1), o);
      }
      case "maxPool2d": case "averagePool2d":
        return b[n.op](x(0), opt(n, ["windowDimensions", "padding", "strides", "dilations"]));
      case "batchNormalization": {
        const o = opt(n, ["epsilon", "axis"]);
        if (n.scale) o.scale = ref(n.scale);
        if (n.bias) o.bias = ref(n.bias);
        return b.batchNormalization(x(0), x(1), x(2), o);
      }
      case "concat": return b.concat(n.inputs.map((_, i) => x(i)), n.axis);
      case "clamp": return b.clamp(x(0), opt(n, ["minValue", "maxValue"]));
      case "hardSigmoid": return b.hardSigmoid(x(0), opt(n, ["alpha", "beta"]));
      case "leakyRelu": return b.leakyRelu(x(0), opt(n, ["alpha"]));
      case "elu": return b.elu(x(0), opt(n, ["alpha"]));
      case "cast": return b.cast(x(0), n.dataType);
    }
    throw new Error(`no builder rule for ${n.op}`);
  }

  const contexts = {};

  window.truestarWebnn = {
    async info(device) {
      const out = {hasML: !!navigator.ml, secure: window.isSecureContext, ua: navigator.userAgent};
      if (!navigator.ml) return out;
      try {
        const ctx = await this.context(device);
        out.context = true;
        if (ctx.opSupportLimits) {
          const l = ctx.opSupportLimits();
          out.preferredInputLayout = l.preferredInputLayout;
          out.ops = Object.keys(l).length;
        }
      } catch (e) { out.error = String(e); }
      return out;
    },

    async context(device) {
      // one context per device type, made again only when the browser reports it lost (making a
      // new context per run exhausted the ML service after a few dozen graphs: "Context is lost")
      const c = contexts[device];
      if (!c || c.isLost) {
        const ctx = await navigator.ml.createContext(device ? {deviceType: device} : {});
        ctx.isLost = false;
        if (ctx.lost) ctx.lost.then((i) => { ctx.isLost = true; ctx.lostWhy = (i && i.message) || "lost"; });
        contexts[device] = ctx;
      }
      return contexts[device];
    },

    async run(device, desc, feeds, repeat) {
      // A lost context (the browser's ML service dropped it) is made again and the graph rerun
      // once; the retry is reported, never hidden.
      try {
        return await this.runOnce(device, desc, feeds, repeat, 0);
      } catch (e) {
        if (!String(e).includes("lost")) throw e;
        if (contexts[device]) contexts[device].isLost = true;
        return await this.runOnce(device, desc, feeds, repeat, 1);
      }
    },

    async runOnce(device, desc, feeds, repeat, retries) {
      const t0 = performance.now();
      const ctx = await this.context(device);
      const b = new MLGraphBuilder(ctx);
      const env = {};
      for (const i of desc.inputs) env[i.name] = b.input(i.name, {dataType: i.dtype, shape: i.shape});
      for (const c of desc.consts) {
        env[c.name] = b.constant({dataType: c.dtype, shape: c.shape}, new TYPED[c.dtype](bytes(c.data)));
      }
      for (const n of desc.nodes) {
        try { env[n.outputs[0]] = emit(b, env, n); }
        catch (e) { throw new Error(`building ${n.type} ${n.name}: ${e}`); }
      }
      const named = {};
      const used = new Set();
      desc.outputs.forEach((o, k) => {
        let v = env[o];
        if (v === undefined) throw new Error(`output ${o} is not produced by the graph`);
        // An input, a constant, or an operand already named as an output gets its own identity:
        // the same operand under two output names made Chrome 158 lose the context at build.
        if (used.has(v) || desc.inputs.some((i) => i.name === o) || desc.consts.some((c) => c.name === o)) v = b.identity(v);
        used.add(v);
        named["o" + k] = v;
      });
      const graph = await b.build(named);
      const tin = {};
      for (const i of desc.inputs) {
        tin[i.name] = await ctx.createTensor({dataType: i.dtype, shape: i.shape, writable: true});
        ctx.writeTensor(tin[i.name], new TYPED[i.dtype](bytes(feeds[i.name])));
      }
      const runs = [];
      for (let r = 0; r <= repeat; r++) {
        const tout = {};
        for (const k of Object.keys(named)) {
          tout[k] = await ctx.createTensor({dataType: named[k].dataType, shape: named[k].shape, readable: true});
        }
        ctx.dispatch(graph, tin, tout);
        const res = [];
        for (const k of Object.keys(named)) {
          res.push({dtype: named[k].dataType, shape: named[k].shape, data: b64(await ctx.readTensor(tout[k]))});
        }
        runs.push(res);
        for (const k of Object.keys(tout)) tout[k].destroy();
      }
      for (const k of Object.keys(tin)) tin[k].destroy();
      graph.destroy();
      return {runs, ms: performance.now() - t0, contextRetries: retries};
    },
  };
})();
