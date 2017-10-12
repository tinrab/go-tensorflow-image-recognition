package main

import (
	"bytes"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func makeTensorFromImage(imageBuffer *bytes.Buffer) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(imageBuffer.String())
	if err != nil {
		return nil, err
	}
	graph, input, output, err := makeNormalizeJPEGImageGraph()
	if err != nil {
		return nil, err
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func makeNormalizeJPEGImageGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		H, W  = 224, 224
		Mean  = float32(117)
		Scale = float32(1)
	)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func makeNormalizePNGImageGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		H, W  = 224, 224
		Mean  = float32(117)
		Scale = float32(1)
	)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodePng(s, input, op.DecodePngChannels(3)), tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}
