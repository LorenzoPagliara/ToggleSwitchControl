import do_mpc
from casadi import *
from ..models import ToggleSwitchUncertainModel


class ControllerMPC:
    """
    Class representing the MPC controller. 
    It contains an instance of the model to be controlled, which it uses to define the cost function and the constraints to be applied. 

    This class defines the parameters of the controller and simulator and is capable of executing a control loop, 
    or making the model execute an input trajectory taken as input. 
    It allows deterministic simulations or in the presence of process disturbances and measurement noise.
    """

    def __init__(self, modelObj, t_step, setup_mpc, stochastic=False) -> None:
        """Constructor of the class, sets the parameters of the controller, simulator and estimator, which will be used in the control loop.

        Args:
            modelObj (ToggleSwitchModel): Instance of the model class to be controlled.
            t_step (float64): Controller sampling time.
            setup_mpc (dict): Dictionary containing controller parameters such as sampling time, time horizon, etc.
            stochastic (bool, optional): Boolean parameter determining whether the control algorithm will be executed deterministically or in the presence of noise and disturbances. Defaults to False.
        """
        self.stochastic = stochastic
        self.t_step = t_step
        self.modelObj = modelObj
        self.controller = self.controller_mpc(setup_mpc)
        self.simulator = self.simulator_mpc()
        self.estimator = do_mpc.estimator.StateFeedback(modelObj.get_model())

    def controller_mpc(self, setup_mpc):
        """It defines the controller parameters, sets the cost function and constraints according to the model to be controlled and returns an instance of the controller.

        Args:
            setup_mpc (dict): Dictionary containing controller parameters such as sampling time, time horizon, etc.

        Returns:
            do_mpc.controller.MPC: Instance of the controller.
        """

        model = self.modelObj.get_model()

        mpc = do_mpc.controller.MPC(model)

        mpc.set_param(**setup_mpc)

        # Cost function
        mpc = self.modelObj.set_cost(mpc)

        # Constraints
        mpc = self.modelObj.set_constraints(mpc)

        if isinstance(self.modelObj, ToggleSwitchUncertainModel):
            mpc = self.modelObj.set_uncertain_parameters(mpc)

        mpc.setup()

        return mpc

    def simulator_mpc(self):
        """Defines the parameters of the simulator and, in the case of a model to be controlled with uncertain parameters, defines the function for updating these parameters at runtime.

        Returns:
            do_mpc.simulator.Simulator: Instance of the simulator.
        """

        simulator = do_mpc.simulator.Simulator(self.modelObj.get_model())
        simulator.set_param(t_step=self.t_step)

        if isinstance(self.modelObj, ToggleSwitchUncertainModel):
            simulator = self.modelObj.uncertain_parameters_function(simulator)

        simulator.setup()

        return simulator

    def control_loop(self, x_0, steps, type, episodes=1):
        """It defines the control loop, allowing it to run for a certain number of episodes and guaranteeing the possibility of adding noise and disturbances. 
        At the end of each episode, it formats and stores the results obtained, using the methods defined by the model class, and exports them to files.

        Args:
            x_0 (numpy.ndarray[float64]): Initial conditions.
            steps (int): Number of control loop steps.
            type (str): Defines the type of model controlled between: deterministic, stochastic, randomic and uncertain.
            episodes (int, optional): Defines the number of repetitions of the control loop. Defaults to 1.
        """

        model = self.modelObj.get_model()

        for e in range(episodes):

            self.estimator.reset_history()
            self.controller.reset_history()
            self.simulator.reset_history()

            self.controller.x0 = x_0
            self.simulator.x0 = x_0
            self.estimator.x0 = x_0

            self.controller.set_initial_guess()

            for k in range(steps):

                if self.stochastic:
                    v0 = np.random.randn(model.n_v, 1)
                    w0 = 0.2*np.random.randn(model.n_w, 1)
                else:
                    v0 = 0*np.random.randn(model.n_v, 1)
                    w0 = 0*np.random.randn(model.n_w, 1)

                u = self.controller.make_step(x_0)

                if (self.controller.t0 - self.t_step) % 15 == 0:
                    ukm1 = u
                else:
                    u = ukm1

                y_next = self.simulator.make_step(u, v0, w0)
                x_0 = self.estimator.make_step(y_next)

            self.modelObj.set_trajectories(self.controller.data)
            self.modelObj.export_results(type, 'results' + str(e), 'w')

    def control_loop_no_constraints(self, x_0, steps, episodes=1):
        """It defines the control loop with no constraints.

        Args:
            x_0 (numpy.ndarray[float64]): Initial conditions.
            steps (int): Number of control loop steps.
            episodes (int, optional): Defines the number of repetitions of the control loop. Defaults to 1.
        """

        model = self.modelObj.get_model()

        for e in range(episodes):

            self.estimator.reset_history()
            self.controller.reset_history()
            self.simulator.reset_history()

            self.controller.x0 = x_0
            self.simulator.x0 = x_0
            self.estimator.x0 = x_0

            self.controller.set_initial_guess()

            for k in range(steps):

                u = self.controller.make_step(x_0)
                y_next = self.simulator.make_step(u)
                x_0 = self.estimator.make_step(y_next)

            self.modelObj.set_trajectories(self.controller.data)

    def execute_trajectory(self, x_0, steps, type, u):
        """Allows the model to execute an input trajectory taken as input. Finally, it formats, stores and exports the results to files.

        Args:
            x_0 (numpy.ndarray[float64]): Initial conditions.
            steps (int): Number of control loop steps.
            type (str): Defines the type of model controlled between: deterministic, stochastic, randomic and uncertain.
            u (numpy.ndarray[float64]): Input trajecory.
        """

        self.simulator.reset_history()

        self.controller.x0 = x_0
        self.simulator.x0 = x_0
        self.estimator.x0 = x_0

        self.controller.set_initial_guess()

        for k in range(steps):

            uk = self.controller.make_step(x_0)
            uk = np.array([[u[k, 0]], [u[k, 1]]])
            y_next = self.simulator.make_step(uk)
            x_0 = self.estimator.make_step(y_next)

        self.modelObj.set_trajectories(self.controller.data)
        self.modelObj.trajectories['inputs']['aTc'] = [u[i, 0] for i in range(steps)]
        self.modelObj.trajectories['inputs']['IPTG'] = [u[i, 1] for i in range(steps)]
        self.modelObj.export_results(type, 'results0', 'w')
