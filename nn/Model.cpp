/*
	Model.cpp

	Class Model
	An Abstract class for a standard neural network template.
	Its purpose is to serve as the base in which will be added layers.

*/

// Importing class file
#include "Model.h"


// Layer adder with parameters definition

void Model::addLayer(Conv layer)
{
    conv_layers.push_back(new Conv(layer));
    layers_order.push_back(0);
}

void Model::addLayer(Pool layer)
{
    pool_layers.push_back(new Pool(layer));
    layers_order.push_back(1);
}

void Model::addLayer(Dense layer)
{
    dense_layers.push_back(new Dense(layer));
    layers_order.push_back(2);
}

void Model::addLayer(Output layer)
{
    output_layer = new Output(layer);
    layers_order.push_back(3);
}


//Constructor definition with parameters
Model::Model(const int nb_kerne):nb_kernel(nb_kerne) {}


//Constructor definition with parameters
Model::Model(const Model& model)
{
    for(Conv* conv:model.conv_layers)
        conv_layers.push_back(new Conv(*conv));

    for(Pool* pool:model.pool_layers)
        pool_layers.push_back(new Pool(*pool));

    for(Dense* dense:model.dense_layers)
        dense_layers.push_back(new Dense(*dense));

    output_layer = new Output(*(model.output_layer));

    layers_order = model.layers_order;
    nb_kernel = model.nb_kernel;
}

// Model's layers initializer definition
void Model::init(const int nb_kernel, const int conv_shape, const int conv_stride, const int pool_shape, const int pool_stride, const int padding)
{
	this->nb_kernel = nb_kernel;

	for(Conv* layer:conv_layers)
        layer->init(nb_kernel, conv_shape, conv_stride, padding);

	for(Pool* layer:pool_layers)
		layer->init(pool_shape, pool_stride);

	for(Dense* layer:dense_layers)
		layer->init("heInit");

	output_layer->init("heInit");

}


// Model's train method definition
void Model::train(const std::vector<std::vector<std::vector<double>>> x_train, const std::vector<std::vector<double>> y_train, const float alpha, const int epochs, int batch_size, const float moment, const float threshold)
{

	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	std::size_t idx;
	int len_x;// iter;
	std::vector<int> available;
	std::vector<double> pred, output, selected_y;

	std::vector<std::vector<double>> inputs_dense, input;
	std::vector<std::vector<std::vector<double>>> prev_grad;
	std::vector<std::vector<std::vector<std::vector<double>>>> inputs_conv;

	len_x = static_cast<int>(x_train.size());

	if(batch_size == 0) batch_size = len_x;

	for(int epoch=1; epoch < epochs; epoch++)
	{
		std::cout << "Epoch: " << epoch << std::endl;

		//iter = len_x / batch_size;
		//if(int(iter) != iter) iter = int(iter) + 1 else iter = int(iter);

		pred = {};
		inputs_conv = {};
		inputs_dense = {};
		output = {};

		for(int i = 0; i < batch_size; i++)
			available.push_back(i);

        cout << "Progress : [";
		selected_y = {};
		for(int record = 0; record < batch_size; record++)
		{
			std::cout << "#";
			//std::cout << conv_layers[0]->kernel[0][0][0] << dense_layers[0]->nodes[0].weights[0] << std::endl;
			if(available.size())
			{
				idx = std::rand() % available.size(); // Select a random number in the range of available
				selected_y = {y_train[available[idx]]};
				input = x_train[available[idx]];

				auto result = forward(input);
				inputs_conv = std::get<0>(result);
				inputs_dense = std::get<1>(result);

				available.erase(available.begin()+idx); // Remove the used index

				pred = inputs_dense.back();
				inputs_dense.pop_back();

                prev_grad = backward(pred, input, inputs_dense, inputs_conv, selected_y, alpha, moment, prev_grad);

			}
		}

		for(int i = 0; i < len_x; i++)
		{
			auto predic = forward(x_train[i]);
            double error = 0;

            for(int j = 0; j < static_cast<int>(std::get<1>(predic).back().size()); j++)
                error += y_train[i][j] - std::get<1>(predic).back()[j];

            output.push_back(error / static_cast<double>(std::get<1>(predic).back().size()));
		}

		double error = MSE(output);
		std::cout << "]\nError: " << error << std::endl;

		if(error <= threshold) break;

	}

	// Freeing space

}


// Model's forward method definition
std::tuple<std::vector<std::vector<std::vector<std::vector<double>>>>, std::vector<std::vector<double>>> Model::forward(vector<vector<double>> x) const
{
	std::vector<int> layer_iter(3, 0);
	std::vector<std::vector<double>> outputs_dense;
	std::vector<std::vector<std::vector<std::vector<double>>>> outputs_conv;

	for(int layer_ord:layers_order)
	{
		if(layer_ord == 2 || layer_ord == 3) // Dense or Output layer
		{
			if(outputs_dense.empty())
			{
				std::vector<double> a, b;
				Flatten flatten;

				for(int i = 0; i < static_cast<int>(outputs_conv.back().size()); i++)
				{
					b = flatten.flat(outputs_conv.back()[i]);
					a.insert(a.end(), b.begin(), b.end());
					b.clear();
				}
				a = min_max(a);
				outputs_dense.push_back(a);
			}
			if(layer_ord == 2)
			{
				outputs_dense.push_back(dense_layers[layer_iter[2]]->forward(outputs_dense.back()));
				layer_iter[2]++;
			}
			else
            {
				outputs_dense.push_back(output_layer->forward(outputs_dense.back()));
            }
		}

		if(layer_ord == 0) // Convolution layer
		{
			if(outputs_conv.empty())
				outputs_conv.push_back(conv_layers[layer_iter[0]]->forward(x));
			else
			{
			    outputs_conv.push_back(conv_layers[layer_iter[0]]->forward(outputs_conv.back()));
				layer_iter[1]++;
			}

			layer_iter[0]++;
		}
		if(layer_ord == 1) // Pooling layer
		{
			try
			{
			    std::vector<std::vector<std::vector<double>>> a = pool_layers[layer_iter[1]]->forward(outputs_conv.back());
                outputs_conv.pop_back();
				outputs_conv.push_back(a);
			}
			catch(...)
			{
				std::cerr << "There should be a convolution layer  before any pooling layer." << std::endl;
			}
		}

	}

	//Remove the output of the last convolution layer
	outputs_conv.pop_back();

	return std::make_tuple(outputs_conv, outputs_dense);
}


// Model's backward method definition
std::vector<vector<vector<double>>> Model::backward(std::vector<double> pred, std::vector<std::vector<double>> input, std::vector<std::vector<double>> outputs_dense, std::vector<std::vector<std::vector<std::vector<double>>>> outputs_conv, const std::vector<double> y_train, const float alpha, const float moment, std::vector<std::vector<std::vector<double>>> prev_grad)
{
	std::vector<double> grad = gradient(pred, y_train);

    //std::cout << "###back " << prev_grad.size() << std::endl;

	std::vector<int> layer_iter(3, 0);
	std::vector<std::vector<std::vector<double>>> out_matrixes, prev_grad_temp;
	std::vector<std::vector<double>> err_mat, prev_grad_;
	std::vector<double> out_gradient;
	std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> out;

	std::vector<Conv*> reversed_conv_layers = reverse(conv_layers);
	std::vector<Dense*> reversed_dense_layers = reverse(dense_layers);

	for(int k = static_cast<int>(layers_order.size()) - 1; k > -1; k--)
	{
	    //std::cout << output_layer->nodes[0].input_len << "###  ###" << std::endl;
	    if(layers_order[k]==2 || layers_order[k]==3) // Dense or Output layer
		{
			if(layers_order[k]==3) // Output layer
			{
                if(prev_grad.empty())
				{
				    out = output_layer->update(grad, alpha, pred, moment);
					err_mat = out.first;
					prev_grad_ = out.second;
				}
				else
				{
					out = output_layer->update(grad, alpha, pred, moment, prev_grad.back());
					prev_grad.pop_back();
					err_mat = out.first;
					prev_grad_ = out.second;
				}
			}
			else // Dense layer
			{
				if(prev_grad.empty())
				{
					out = reversed_dense_layers[layer_iter[2]]->update(out_gradient, alpha, outputs_dense[outputs_dense.size()-2], outputs_dense.back(), moment);
					outputs_dense.pop_back();
					layer_iter[2]++;
					err_mat = out.first;
					prev_grad_ = out.second;
				}
				else
				{
					out = reversed_dense_layers[layer_iter[2]]->update(out_gradient, alpha, outputs_dense[outputs_dense.size()-2], outputs_dense.back(), moment, prev_grad.back());
					outputs_dense.pop_back();
					prev_grad.pop_back();
					layer_iter[2]++;
					err_mat = out.first;
					prev_grad_ = out.second;
				}
			}

			prev_grad_temp.push_back(prev_grad_);

			int n1 = static_cast<int>(err_mat.size());
			int n2 = static_cast<int>(err_mat[0].size());

			out_gradient.resize(n2);
			out_gradient.assign(n2, 0);

			for(int i = 0; i < n1; i++)
			{
				for(int j=0; j < n2; j++)
					out_gradient[j] += err_mat[i][j];
			}

			if(layers_order[k-1] == 0) // Conv layer comes next
			{
			    //inputs_dense.pop_back();
				int n = static_cast<int>(sqrt(n2 / nb_kernel));
				out_matrixes = reshape(out_gradient, nb_kernel, n);
			}
		}
		if(layers_order[k] == 0) // Conv layer
		{
			if(k > 2)
			{
                out_matrixes = reversed_conv_layers[layer_iter[0]]->update(out_matrixes, outputs_conv.back(), outputs_conv[outputs_conv.size()-2], alpha);
				outputs_conv.pop_back();
			}
			else if(k == 2)
			{
				out_matrixes = reversed_conv_layers[layer_iter[0]]->update(out_matrixes, outputs_conv.back(), input, alpha);
				outputs_conv.pop_back();
			}
			else
				reversed_conv_layers[layer_iter[0]]->update(out_matrixes, input, alpha);
			layer_iter[0]++;
		}

	}


    //std::cout << "###back1 " << prev_grad.size() << std::endl;

	return reverse(prev_grad_temp);

}




std::vector<double> Model::predict(const std::vector<std::vector<double>> x) const // Model's predict method declaration
{
    std::vector<double> input;
    std::vector<int> layer_iter(3, 0);
	std::vector<std::vector<double>> outputs_dense;
	std::vector<std::vector<std::vector<std::vector<double>>>> outputs_conv;

	for(int layer_ord:layers_order)
	{
		if(layer_ord == 2 || layer_ord == 3) // Dense or Output layer
		{
			if(outputs_dense.empty())
			{
				std::vector<double> a, b;
				Flatten flatten;

				for(int i = 0; i < static_cast<int>(outputs_conv.back().size()); i++)
				{
					b = flatten.flat(outputs_conv.back()[i]);
					a.insert(a.end(), b.begin(), b.end());
					b.clear();
				}
				a = min_max(a);
				outputs_dense.push_back(a);
			}
			if(layer_ord == 2)
			{
				outputs_dense.push_back(dense_layers[layer_iter[2]]->forward(outputs_dense.back()));
				layer_iter[2]++;
			}
			else
            {
				outputs_dense.push_back(output_layer->forward(outputs_dense.back()));
            }
		}

		if(layer_ord == 0) // Convolution layer
		{
			if(outputs_conv.empty())
				outputs_conv.push_back(conv_layers[layer_iter[0]]->forward(x));
			else
			{
			    outputs_conv.push_back(conv_layers[layer_iter[0]]->forward(outputs_conv.back()));
				layer_iter[1]++;
			}

			layer_iter[0]++;
		}
		if(layer_ord == 1) // Pooling layer
		{
			try
			{
			    std::vector<std::vector<std::vector<double>>> a = pool_layers[layer_iter[1]]->forward(outputs_conv.back());
                outputs_conv.pop_back();
				outputs_conv.push_back(a);
			}
			catch(...)
			{
				std::cerr << "There should be a convolution layer  before any pooling layer." << std::endl;
			}
		}

	}

	//Remove the output of the last convolution layer
	//outputs_conv.pop_back();
	/*
	for( obj:std::input)
		obj.clear();
	input.clear();

	a.clear();
	b.clear();

	*/

	for (size_t i = 0; i < input.size(); i++)
	{
		if(input[i] >= 0.5)
			input[i] = 1;
		else
			input[i] = 0;
	}

	return input;
}




Model::~Model()
{
    for(Conv* obj : conv_layers)
        delete obj;
	conv_layers.clear();

	for(Pool* obj : pool_layers)
        delete obj;
	pool_layers.clear();

	for(Dense* obj : dense_layers)
        delete obj;
	dense_layers.clear();

	delete output_layer;
}

