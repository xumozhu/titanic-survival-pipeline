
# This project uses only the R packages included with the base R installation.
# No additional CRAN packages are required for the Titanic logistic regression model.

# Display the installed R version for reproducibility.
cat("R Version:\n")
cat(R.version.string, "\n")

# Confirm that the built-in stats package is available for glm().
cat("\nStats Package Version:\n")
print(packageVersion("stats"))

# Confirm that no external packages need to be installed.
cat("\nAll required R packages are available.\n")
