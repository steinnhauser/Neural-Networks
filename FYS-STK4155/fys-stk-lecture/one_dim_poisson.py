import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx("float64") # This is good


# Define analytic solution
def g_analytic(x):
    return x * (1 - x) * tf.exp(x)

# Define grid
num_points = 11
# domain / boundaries:
start = tf.constant(0, dtype = tf.float64) # force it to be 64 bit.
stop = tf.constant(1, dtype = tf.float64)

x = tf.reshape(
    tf.linspace(start, stop, num_points),
    (-1,1)
)

# Define model (e.g. sequential model from keras)
# Make it a subclass of the keras model.
class DNModel(tf.keras.Model):
    def __init__(self):
        super(DNModel, self).__init__() #pass to parent class

        self.dense_1 = tf.keras.layers.Dense(20, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense_1(inputs) # returns a value x
        x = self.dense_2(x) # thread inputs through each layer manually

        return self.out(x)

# Define right-hand side solution
@tf.function # tf can treat this function as a tensorflow function (speeds up)
def rhs(x):
    return (3 * x + x ** 2) * tf.exp(x)

# Define trial solution
@tf.function
def trial_solution(model, x):
    return x * (1 - x) * model(x) # neural network evaluated at x.

# Define loss function
@tf.function
def loss(model, x):
    # since we're doing something that's "both analytic and numeric at the same time":
    # need to specify gradient tape:
    with tf.GradientTape() as tape:
        tape.watch(x) # treat x as a symbolic value.
        # diff twice, have to specify one more:
        with tf.GradientTape() as tape2:
            tape2.watch(x)

            trial = trial_solution(model, x)

        d_trial = tape2.gradient(trial, x)
    d2_trial = tape.gradient(d_trial, x)

    return tf.losses.MSE(tf.zeros_like(d2_trial), -d2_trial-rhs(x)) # takes in a 'true' and a predicted value

# Define gradient method
@tf.function
def grad(model, x):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)
    # return loss value and the gradient of the loss value w.r.t the NN.
    # tape.gradient(loss_value, model.trainable_variables) is essentially the back propogation

# Initialize model and optimizer
model = DNModel()
optimizer = tf.keras.optimizers.Adam(0.01) #learning rate.

# Run training loop
# Apply gradients in optimizer
# Output loss improvement
num_epochs = 2000
for epoch in range(num_epochs):
    cost, gradients = grad(model, x)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(
        f"Step: {optimizer.iterations.numpy()}, "
        + f"Loss: {tf.reduce_mean(cost.numpy())}"
    )

# Plot solution on larger grid
x = tf.reshape(tf.linspace(start, stop, 1001), (-1,1))

plt.plot(x, trial_solution(model, x), label="Neural")
plt.plot(x, g_analytic(x), label="Analytic")
plt.legend()
plt.show()
