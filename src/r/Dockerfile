
# ==== STEP 1: SET UP THE R ENVIRONMENT ====

# Use a fixed R version to keep the runtime environment reproducible.
FROM rocker/r-ver:4.5.2

# Set the working directory inside the container.
WORKDIR /app

# ==== STEP 2: CHECK R PACKAGE DEPENDENCIES ====

# Copy the R dependency setup script into the container.
COPY src/r/install_packages.R src/r/install_packages.R

# Run the dependency script during the Docker image build.
RUN Rscript src/r/install_packages.R

# ==== STEP 3: COPY THE R MODEL ====

# Copy the Titanic logistic regression script into the container.
COPY src/r/titanic_model.R src/r/titanic_model.R

# Create a directory for the Titanic CSV datasets.
RUN mkdir -p src/data

# ==== STEP 4: RUN THE R MODEL ====

# Execute the Titanic model when the container starts.
CMD ["Rscript", "src/r/titanic_model.R"]
