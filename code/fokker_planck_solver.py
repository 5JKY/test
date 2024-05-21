import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve


class FokkerPlanckSolover_ReAb:
    # Crank–Nicolson type is the default numerical method
    def __init__(
        self,
        ht,
        hx,
        x_arr,
        u0_arr,
        beta_Up_arr,
        beta_Upp_arr,
        D_arr,
        phi=1 / 2,
        gamma=1 / 2,
    ):
        self.hx = hx
        self.ht = ht
        self.x_arr = x_arr
        self.u0_arr = u0_arr
        self.beta_Up_arr = beta_Up_arr
        self.beta_Upp_arr = beta_Upp_arr
        self.D_arr = D_arr
        self.phi = phi
        self.gamma = gamma
        self.Ph_mat, self.Qh_mat = self.assemble_matrices()

        if self.phi == 1 / 2 and self.gamma == 1 / 2:
            self.method_name = "Crank–Nicolson"
        elif self.phi == 0 and self.gamma == 1 / 2:
            self.method_name = "ETCS"
        elif self.phi == 0 and self.gamma == 1:
            self.method_name = "ETBS"
        elif self.phi == 0 and self.gamma == 0:
            self.method_name = "ETFS (upwind explicit)"
        elif self.phi == 1 and self.gamma == 1 / 2:
            self.method_name = "ITCS"
        elif self.phi == 1 and self.gamma == 1:
            self.method_name = "ITBS"
        elif self.phi == 1 and self.gamma == 0:
            self.method_name = "ITFS(upwind implicit)"
        else:
            self.method_name = "Other"

    def assemble_matrices(self):
        # a(x), alpha, b(x)
        a_arr = -self.D_arr * self.beta_Up_arr
        alpha_arr = self.D_arr
        b_arr = self.D_arr * self.beta_Upp_arr
        # c, s
        c_arr = self.ht / self.hx * a_arr
        s_arr = self.ht / (self.hx**2) * alpha_arr

        # coefficients on the left side (Ph_mat[i])
        A0_arr = -self.phi * (s_arr + c_arr * (1 - self.gamma))
        A1_arr = np.ones(self.x_arr.size) - self.phi * (
            self.ht * b_arr + (2 * self.gamma - 1) * c_arr - 2 * s_arr
        )
        A2_arr = self.phi * (self.gamma * c_arr - s_arr)
        # coefficients on the right side (Qh_mat[i])
        A3_arr = (1 - self.phi) * (c_arr * (1 - self.gamma) + s_arr)
        A4_arr = np.ones(self.x_arr.size) + (1 - self.phi) * (
            self.ht * b_arr + (2 * self.gamma - 1) * c_arr - 2 * s_arr
        )
        A5_arr = (1 - self.phi) * (s_arr - self.gamma * c_arr)

        # Assemble coefficient matrix Ph for u^[n+1]
        Ph_mat = (
            np.diag(A0_arr[1:-1], k=-1)
            + np.diag(A1_arr[0:-1], k=0)
            + np.diag(A2_arr[0:-2], k=1)
        )
        # Apply reflecting boundary condition
        Ph_mat[0, 0] -= 2 * self.hx * a_arr[0] / alpha_arr[0] * A0_arr[0]
        Ph_mat[0, 1] += A0_arr[0]

        # Assemble coefficient matrix Qh for u^n
        Qh_mat = (
            np.diag(A3_arr[1:-1], k=-1)
            + np.diag(A4_arr[0:-1], k=0)
            + np.diag(A5_arr[0:-2], k=1)
        )
        # Apply reflecting boundary condition
        Qh_mat[0, 0] -= 2 * self.hx * a_arr[0] / alpha_arr[0] * A3_arr[0]
        Qh_mat[0, 1] += A3_arr[0]

        return Ph_mat, Qh_mat

    def solve(self, Tf):
        u_arr = np.copy(self.u0_arr)
        # Apply absorbing boundary condition
        u_arr[-1] = 0

        Nt = int(Tf / self.ht)
        for _ in range(1, Nt + 1):
            # Solve the equation system Ph@u^[n+1] = Qh@u^n
            # u_arr[0:-1] = np.linalg.solve(self.Ph_mat, self.Qh_mat@u_arr[0:-1])
            u_arr[0:-1] = spsolve(self.Ph_mat, self.Qh_mat @ u_arr[0:-1])

        return u_arr

    def plot_solution(self, Tf, freq=10):
        u_arr = np.copy(self.u0_arr)
        # Apply absorbing boundary condition
        u_arr[-1] = 0

        Nt = int(Tf / self.ht)
        interval = int(Nt / freq)
        count = 1

        for n in range(1, Nt + 1):
            # Solve the equation system Ph@u^[n+1] = Qh@u^n
            # u_arr[0:-1] = np.linalg.solve(self.Ph_mat, self.Qh_mat@u_arr[0:-1])
            u_arr[0:-1] = spsolve(self.Ph_mat, self.Qh_mat @ u_arr[0:-1])

            # Plot the solution at every 10 time steps
            if n == count * interval:
                plt.plot(self.x_arr, u_arr, label=f"t={n*self.ht:.2f}")
                count += 1

        # Plot formatting
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"{self.method_name} Method solve FPE (ReAb)")
        plt.legend()
        plt.grid(True)
        plt.show()
        return f"number of iterations is {interval*(count-1)}"


class FokkerPlanckSolover_AbRe:
    # Crank–Nicolson type is the default numerical method
    def __init__(
        self,
        ht,
        hx,
        x_arr,
        u0_arr,
        beta_Up_arr,
        beta_Upp_arr,
        D_arr,
        phi=1 / 2,
        gamma=1 / 2,
    ):
        self.hx = hx
        self.ht = ht
        self.x_arr = x_arr
        self.u0_arr = u0_arr
        self.beta_Up_arr = beta_Up_arr
        self.beta_Upp_arr = beta_Upp_arr
        self.D_arr = D_arr
        self.phi = phi
        self.gamma = gamma
        self.Ph_mat, self.Qh_mat = self.assemble_matrices()

        if self.phi == 1 / 2 and self.gamma == 1 / 2:
            self.method_name = "Crank–Nicolson"
        elif self.phi == 0 and self.gamma == 1 / 2:
            self.method_name = "ETCS"
        elif self.phi == 0 and self.gamma == 1:
            self.method_name = "ETBS"
        elif self.phi == 0 and self.gamma == 0:
            self.method_name = "ETFS (upwind explicit)"
        elif self.phi == 1 and self.gamma == 1 / 2:
            self.method_name = "ITCS"
        elif self.phi == 1 and self.gamma == 1:
            self.method_name = "ITBS"
        elif self.phi == 1 and self.gamma == 0:
            self.method_name = "ITFS(upwind implicit)"
        else:
            self.method_name = "Other"

    def assemble_matrices(self):
        # a(x), alpha, b(x)
        a_arr = -self.D_arr * self.beta_Up_arr
        alpha_arr = self.D_arr
        b_arr = self.D_arr * self.beta_Upp_arr
        # c, s
        c_arr = self.ht / self.hx * a_arr
        s_arr = self.ht / (self.hx**2) * alpha_arr

        # coefficients on the left side (Ph_mat[i])
        A0_arr = -self.phi * (s_arr + c_arr * (1 - self.gamma))
        A1_arr = np.ones(self.x_arr.size) - self.phi * (
            self.ht * b_arr + (2 * self.gamma - 1) * c_arr - 2 * s_arr
        )
        A2_arr = self.phi * (self.gamma * c_arr - s_arr)
        # coefficients on the right side (Qh_mat[i])
        A3_arr = (1 - self.phi) * (c_arr * (1 - self.gamma) + s_arr)
        A4_arr = np.ones(self.x_arr.size) + (1 - self.phi) * (
            self.ht * b_arr + (2 * self.gamma - 1) * c_arr - 2 * s_arr
        )
        A5_arr = (1 - self.phi) * (s_arr - self.gamma * c_arr)

        # Assemble coefficient matrix Ph for u^[n+1]
        Ph_mat = (
            np.diag(A0_arr[2:], k=-1)
            + np.diag(A1_arr[1:], k=0)
            + np.diag(A2_arr[1:-1], k=1)
        )
        # Apply reflecting boundary condition
        Ph_mat[-1, -1] += 2 * self.hx * a_arr[-1] / alpha_arr[-1] * A2_arr[-1]
        Ph_mat[-1, -2] += A2_arr[-1]

        # Assemble coefficient matrix Qh for u^n
        Qh_mat = (
            np.diag(A3_arr[2:], k=-1)
            + np.diag(A4_arr[1:], k=0)
            + np.diag(A5_arr[1:-1], k=1)
        )
        # Apply reflecting boundary condition
        Qh_mat[-1, -1] += 2 * self.hx * a_arr[-1] / alpha_arr[-1] * A5_arr[-1]
        Qh_mat[-1, -2] += A5_arr[-1]

        return Ph_mat, Qh_mat

    def solve(self, Tf):
        u_arr = np.copy(self.u0_arr)
        # Apply absorbing boundary conditions
        u_arr[0] = 0

        Nt = int(Tf / self.ht)
        for _ in range(1, Nt + 1):
            # Solve the equation system Ph@u^[n+1] = Qh@u^n
            # u_arr[1:] = np.linalg.solve(self.Ph_mat, self.Qh_mat@u_arr[1:])
            u_arr[1:] = spsolve(self.Ph_mat, self.Qh_mat @ u_arr[1:])

        return u_arr

    def plot_solution(self, Tf, freq=10):
        u_arr = np.copy(self.u0_arr)
        # Apply absorbing boundary conditions
        u_arr[0] = 0

        Nt = int(Tf / self.ht)
        interval = int(Nt / freq)
        count = 1

        for n in range(1, Nt + 1):
            # Solve the equation system Ph@u^[n+1] = Qh@u^n
            # u_arr[1:] = np.linalg.solve(self.Ph_mat, self.Qh_mat@u_arr[1:])
            u_arr[1:] = spsolve(self.Ph_mat, self.Qh_mat @ u_arr[1:])

            # Plot the solution at every 10 time steps
            if n == count * interval:
                plt.plot(self.x_arr, u_arr, label=f"t={n*self.ht:.2f}")
                count += 1

        # Plot formatting
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"{self.method_name} Method solve FPE (AbRe)")
        plt.legend()
        plt.grid(True)
        plt.show()
        return f"number of iterations is {interval*(count-1)}"


class FokkerPlanckSolover_AbAb:
    # Crank–Nicolson type is the default numerical method
    def __init__(
        self,
        ht,
        hx,
        x_arr,
        u0_arr,
        beta_Up_arr,
        beta_Upp_arr,
        D_arr,
        phi=1 / 2,
        gamma=1 / 2,
    ):
        self.hx = hx
        self.ht = ht
        self.x_arr = x_arr
        self.u0_arr = u0_arr
        self.beta_Up_arr = beta_Up_arr
        self.beta_Upp_arr = beta_Upp_arr
        self.D_arr = D_arr
        self.phi = phi
        self.gamma = gamma
        self.Ph_mat, self.Qh_mat = self.assemble_matrices()

        if self.phi == 1 / 2 and self.gamma == 1 / 2:
            self.method_name = "Crank–Nicolson"
        elif self.phi == 0 and self.gamma == 1 / 2:
            self.method_name = "ETCS"
        elif self.phi == 0 and self.gamma == 1:
            self.method_name = "ETBS"
        elif self.phi == 0 and self.gamma == 0:
            self.method_name = "ETFS (upwind explicit)"
        elif self.phi == 1 and self.gamma == 1 / 2:
            self.method_name = "ITCS"
        elif self.phi == 1 and self.gamma == 1:
            self.method_name = "ITBS"
        elif self.phi == 1 and self.gamma == 0:
            self.method_name = "ITFS(upwind implicit)"
        else:
            self.method_name = "Other"

    def assemble_matrices(self):
        # a(x), alpha, b(x)
        a_arr = -self.D_arr * self.beta_Up_arr
        alpha_arr = self.D_arr
        b_arr = self.D_arr * self.beta_Upp_arr
        # c, s
        c_arr = self.ht / self.hx * a_arr
        s_arr = self.ht / (self.hx**2) * alpha_arr

        # coefficients on the left side (Ph_mat[i])
        A0_arr = -self.phi * (s_arr + c_arr * (1 - self.gamma))
        A1_arr = np.ones(self.x_arr.size) - self.phi * (
            self.ht * b_arr + (2 * self.gamma - 1) * c_arr - 2 * s_arr
        )
        A2_arr = self.phi * (self.gamma * c_arr - s_arr)
        # coefficients on the right side (Qh_mat[i])
        A3_arr = (1 - self.phi) * (c_arr * (1 - self.gamma) + s_arr)
        A4_arr = np.ones(self.x_arr.size) + (1 - self.phi) * (
            self.ht * b_arr + (2 * self.gamma - 1) * c_arr - 2 * s_arr
        )
        A5_arr = (1 - self.phi) * (s_arr - self.gamma * c_arr)

        # Assemble coefficient matrix Ph for u^[n+1]
        Ph_mat = (
            np.diag(A0_arr[2:-1], k=-1)
            + np.diag(A1_arr[1:-1], k=0)
            + np.diag(A2_arr[1:-2], k=1)
        )
        # Assemble coefficient matrix Qh for u^n
        Qh_mat = (
            np.diag(A3_arr[2:-1], k=-1)
            + np.diag(A4_arr[1:-1], k=0)
            + np.diag(A5_arr[1:-2], k=1)
        )

        return Ph_mat, Qh_mat

    def solve(self, Tf):
        u_arr = np.copy(self.u0_arr)
        # Apply absorbing boundary conditions
        u_arr[0] = 0
        u_arr[-1] = 0

        Nt = int(Tf / self.ht)
        for _ in range(1, Nt + 1):
            # Solve the equation system Ph@u^[n+1] = Qh@u^n
            # u_arr[1:-1] = np.linalg.solve(self.Ph_mat, self.Qh_mat@u_arr[1:-1])
            u_arr[1:-1] = spsolve(self.Ph_mat, self.Qh_mat @ u_arr[1:-1])

        return u_arr

    def plot_solution(self, Tf, freq=10):
        u_arr = np.copy(self.u0_arr)
        # Apply absorbing boundary conditions
        u_arr[0] = 0
        u_arr[-1] = 0

        Nt = int(Tf / self.ht)
        interval = int(Nt / freq)
        count = 1

        for n in range(1, Nt + 1):
            # Solve the equation system Ph@u^[n+1] = Qh@u^n
            # u_arr[1:-1] = np.linalg.solve(self.Ph_mat, self.Qh_mat@u_arr[1:-1])
            u_arr[1:-1] = spsolve(self.Ph_mat, self.Qh_mat @ u_arr[1:-1])

            # Plot the solution at every 10 time steps
            if n == count * interval:
                plt.plot(self.x_arr, u_arr, label=f"t={n*self.ht:.2f}")
                count += 1

        # Plot formatting
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"{self.method_name} Method solve FPE (AbAb)")
        plt.legend()
        plt.grid(True)

        plt.show()
        return f"number of iterations is {interval*(count-1)}"


class FokkerPlanckSolover_ReRe:
    # Crank–Nicolson type is the default numerical method
    def __init__(
        self,
        ht,
        hx,
        x_arr,
        u0_arr,
        beta_Up_arr,
        beta_Upp_arr,
        D_arr,
        phi=1 / 2,
        gamma=1 / 2,
    ):
        self.hx = hx
        self.ht = ht
        self.x_arr = x_arr
        self.u0_arr = u0_arr
        self.beta_Up_arr = beta_Up_arr
        self.beta_Upp_arr = beta_Upp_arr
        self.D_arr = D_arr
        self.phi = phi
        self.gamma = gamma
        self.Ph_mat, self.Qh_mat = self.assemble_matrices()

        if self.phi == 1 / 2 and self.gamma == 1 / 2:
            self.method_name = "Crank–Nicolson"
        elif self.phi == 0 and self.gamma == 1 / 2:
            self.method_name = "ETCS"
        elif self.phi == 0 and self.gamma == 1:
            self.method_name = "ETBS"
        elif self.phi == 0 and self.gamma == 0:
            self.method_name = "ETFS (upwind explicit)"
        elif self.phi == 1 and self.gamma == 1 / 2:
            self.method_name = "ITCS"
        elif self.phi == 1 and self.gamma == 1:
            self.method_name = "ITBS"
        elif self.phi == 1 and self.gamma == 0:
            self.method_name = "ITFS(upwind implicit)"
        else:
            self.method_name = "Other"

    def assemble_matrices(self):
        # a(x), alpha, b(x)
        a_arr = -self.D_arr * self.beta_Up_arr
        alpha_arr = self.D_arr
        b_arr = self.D_arr * self.beta_Upp_arr
        # c, s
        c_arr = self.ht / self.hx * a_arr
        s_arr = self.ht / (self.hx**2) * alpha_arr

        # coefficients on the left side (Ph_mat[i])
        A0_arr = -self.phi * (s_arr + c_arr * (1 - self.gamma))
        A1_arr = np.ones(self.x_arr.size) - self.phi * (
            self.ht * b_arr + (2 * self.gamma - 1) * c_arr - 2 * s_arr
        )
        A2_arr = self.phi * (self.gamma * c_arr - s_arr)
        # coefficients on the right side (Qh_mat[i])
        A3_arr = (1 - self.phi) * (c_arr * (1 - self.gamma) + s_arr)
        A4_arr = np.ones(self.x_arr.size) + (1 - self.phi) * (
            self.ht * b_arr + (2 * self.gamma - 1) * c_arr - 2 * s_arr
        )
        A5_arr = (1 - self.phi) * (s_arr - self.gamma * c_arr)

        # Assemble coefficient matrix Ph for u^[n+1]
        Ph_mat = (
            np.diag(A0_arr[1:], k=-1)
            + np.diag(A1_arr[0:], k=0)
            + np.diag(A2_arr[0:-1], k=1)
        )
        # Apply reflecting boundary condition
        Ph_mat[0, 0] -= 2 * self.hx * a_arr[0] / alpha_arr[0] * A0_arr[0]
        Ph_mat[0, 1] += A0_arr[0]
        Ph_mat[-1, -1] += 2 * self.hx * a_arr[-1] / alpha_arr[-1] * A2_arr[-1]
        Ph_mat[-1, -2] += A2_arr[-1]

        # Assemble coefficient matrix Qh for u^n
        Qh_mat = (
            np.diag(A3_arr[1:], k=-1)
            + np.diag(A4_arr[0:], k=0)
            + np.diag(A5_arr[0:-1], k=1)
        )
        # Apply reflecting boundary condition
        Qh_mat[0, 0] -= 2 * self.hx * a_arr[0] / alpha_arr[0] * A3_arr[0]
        Qh_mat[0, 1] += A3_arr[0]
        Qh_mat[-1, -1] += 2 * self.hx * a_arr[-1] / alpha_arr[-1] * A5_arr[-1]
        Qh_mat[-1, -2] += A5_arr[-1]

        return Ph_mat, Qh_mat

    def solve(self, Tf):
        u_arr = np.copy(self.u0_arr)
        Nt = int(Tf / self.ht)
        for _ in range(1, Nt + 1):
            # Solve the equation system Ph@u^[n+1] = Qh@u^n
            # u_arr = np.linalg.solve(self.Ph_mat, self.Qh_mat@u_arr)
            u_arr = spsolve(self.Ph_mat, self.Qh_mat @ u_arr)

        return u_arr

    def plot_solution(self, Tf, freq=10):
        u_arr = np.copy(self.u0_arr)
        Nt = int(Tf / self.ht)
        interval = int(Nt / freq)
        count = 1

        for n in range(1, Nt + 1):
            # Solve the equation system Ph@u^[n+1] = Qh@u^n
            # u_arr = np.linalg.solve(self.Ph_mat, self.Qh_mat@u_arr)
            u_arr = spsolve(self.Ph_mat, self.Qh_mat @ u_arr)

            # Plot the solution at every 10 time steps
            if n == count * interval:
                plt.plot(self.x_arr, u_arr, label=f"t={n*self.ht:.2f}")
                count += 1

        # Plot formatting
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"{self.method_name} Method solve FPE (ReRe)")
        plt.legend()
        plt.grid(True)
        plt.show()
        return f"number of iterations is {interval*(count-1)}"
