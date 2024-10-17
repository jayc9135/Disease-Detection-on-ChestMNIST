## Setup

### 1. Set up a Virtual Environment

First, create and activate a virtual environment to manage dependencies cleanly:

* **Create the virtual environment** :

  ```

  ```
* **Activate the virtual environment** :

  * **Windows** :

    ```
    python -m venv venv
    ```
  * **Linux/Mac** :

    ```
    source venv/bin/activate

    ```

This ensures that the dependencies you install and use are isolated from your global Python environment.

### 2. Install Dependencies

Once the virtual environment is active, install the required dependencies using the `requirements.txt` file:

```
pip install -r requirements.txt
```

This will set up all necessary libraries needed for federated learning and model training.

## Execution

### 3. Run the Program

#### a. Running the Server

The server coordinates the communication between clients and aggregates their models. To start the server:

```
python -m src.server <y or n depending on attack>
```

* Use `y` if you want to include a **label-flipping attack** (poisoned clients) in the training process. The server will store metrics in the `visualization_data/attack_output` directory.
* Use `n` if no attack is applied. The server will store metrics in the `visualization_data/normal_output` directory.

This distinction helps in organizing and analyzing results efficiently.

#### b. Running the Clients

Each client trains on its local dataset. You should run multiple clients (10, for example) in separate terminal windows:

```
python -m src.client <client_id> <poisoned>
```

* `<client_id>`: The ID of the client (e.g., `1`, `2`, etc.).
* `<poisoned>`: Optional; use `poisoned` to indicate if the client is part of the attack.

 **Examples** :

* For a poisoned client:
  ```
  python -m src.client 1 poisoned
  ```
* For a non-poisoned client:
  ```
  python -m src.client 2
  ```

Each client will train on its local data and communicate updates back to the server.

#### c. Visualizing Results

To visualize the metrics collected during the training process:

```
python -m src.visualization

```
