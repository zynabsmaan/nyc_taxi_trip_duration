FROM demisto/sklearn:1.0.0.104162
WORKDIR .
COPY . . 
# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


CMD ["python3", "modeling.py", "--data_version", "1", "--ridge_alpha", "100", "--poly_degree", "2"]