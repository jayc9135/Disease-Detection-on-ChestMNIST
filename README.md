# Federated Learning Setup and Execution

## Setup

### 1. Set up a Virtual Environment

* **Create the virtual environment**

```shell
     python -m venv venv
```

- **Activate the virtual environment**:
  - **Windows**:
    ```shell
    venv\Scripts\activate
    ```
  - **Linux/Mac**:
    ```shell
    source venv/bin/activate
    ```

### 2. Install Dependencies

```shell
   pip install -r requirements.txt
```

### 3. Run the Program

- **Run the server** in a terminal window:

  ```shell
  python server.py
  ```
- **Run 10 clients** in individual terminal windows:

  ```shell
  python client.py <client_id> <poisoned>
  ```
  **Example:**

  - For a poisoned client:
    ```shell
    python client.py 1 poisoned
    ```
  - For a non-poisoned client:
    ```shell
    python client.py 2
    ```
