# Liquid: Transform Your Transformers into Liquid Transformers 💦 💧 

`Liquid` is an open-source Python library that enables you to transform vanilla transformer models into Liquid transformers. Liquid transformers build upon recurrent neural networks, providing a dynamic time constant that evolves over time. This results in improved expressivity and stability in time-series prediction tasks compared to traditional transformer models.

## Benefits of Liquid Transformers

1. **Dynamic Time Constants**: Liquid transformers can adapt to varying time constants in time-series data, making them more flexible and robust in handling temporal patterns.
2. **Superior Expressivity**: Liquid transformers exhibit superior expressivity within the family of neural ordinary differential equations, allowing them to capture complex relationships in the data.
3. **Stable and Bounded Behavior**: Liquid transformers demonstrate stable and bounded dynamics, ensuring consistent performance during training and inference.
4. **Improved Performance**: Liquid transformers have been shown to outperform classical and modern RNNs on time-series prediction tasks.

## Getting Started

### Installation

You can install Liquid using pip:

```
pip install liquid-transformers
```

### Usage 🚀 

To use Liquid, simply import the `apply_liquid()` function and provide the name of your desired pre-trained transformer model. The function will return a Liquid transformer model, ready for training or inference.

```python
from liquid import apply_liquid

model_name = "gpt2"
liquid_gpt2 = apply_liquid(model_name)
```

### Customization 🤖 

You can customize the Liquid parameters by passing them as arguments to the `apply_liquid()` function:

```python
liquid_gpt2 = apply_liquid(model_name, time_constant=1.0, num_steps=10, step_size=0.1)
```

## Roadmap 📖 

We have an ambitious roadmap to advance the Liquid module and make it even more powerful:

1. **Compatibility with Other Transformer Implementations**: Expand compatibility to other popular transformer architectures, such as BERT, RoBERTa, and T5.
2. **Hyperparameter Optimization**: Develop a systematic approach for optimizing Liquid parameters to achieve optimal performance on a given task.
3. **Integration with AutoML Libraries**: Integrate Liquid with popular AutoML libraries, such as Optuna, for automatic hyperparameter optimization.
4. **Support for Additional Frameworks**: Extend the implementation to support other deep learning frameworks, such as TensorFlow and Jax.
5. **Advanced Liquid Architectures**: Explore more advanced Liquid architectures and components to improve performance and applicability to a wider range of tasks.
6. **Multimodal Liquid Transformers**: Investigate the use of Liquid transformers for multimodal tasks, such as video captioning and audio processing.

We welcome contributions from the community to help us achieve these goals and make Liquid an indispensable tool for researchers and practitioners alike. Together, let's build the future of transformer models with Liquid!

## License

Liquid is open-source and licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
