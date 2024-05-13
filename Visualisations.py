Here's an example code snippet for each of the visualizations you mentioned:

1. **Multivariate Data**:
   - Parallel Coordinates Plot:
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt
     from pandas.plotting import parallel_coordinates

     # Sample data
     data = {'Category': ['A', 'B', 'C', 'D'],
             'Feature1': [10, 20, 15, 25],
             'Feature2': [30, 25, 35, 20],
             'Feature3': [15, 10, 5, 20]}
     df = pd.DataFrame(data)

     # Plot parallel coordinates
     plt.figure(figsize=(8, 6))
     parallel_coordinates(df, 'Category', colormap='viridis')
     plt.title('Parallel Coordinates Plot')
     plt.xlabel('Features')
     plt.ylabel('Values')
     plt.legend(loc='upper right')
     plt.show()
     ```

   - Scatterplot Matrix:
     ```python
     import seaborn as sns
     from sklearn.datasets import load_iris

     # Load iris dataset
     iris = load_iris()
     df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
     df['target'] = iris.target

     # Plot scatterplot matrix
     sns.pairplot(df, hue='target', palette='viridis')
     plt.title('Scatterplot Matrix')
     plt.show()
     ```

   - Bubble Charts:
     ```python
     import matplotlib.pyplot as plt
     import numpy as np

     # Sample data
     x = np.random.rand(50)
     y = np.random.rand(50)
     z = np.random.rand(50) * 1000  # Bubble size

     # Plot bubble chart
     plt.scatter(x, y, s=z, alpha=0.5, cmap='viridis')
     plt.title('Bubble Chart')
     plt.xlabel('X-axis')
     plt.ylabel('Y-axis')
     plt.colorbar(label='Bubble Size')
     plt.show()
     ```

   - Radar Charts:
     ```python
     import matplotlib.pyplot as plt
     import numpy as np

     # Sample data
     labels = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
     values = [10, 20, 15, 25]

     # Plot radar chart
     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
     values += values[:1]
     angles += angles[:1]

     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
     ax.fill(angles, values, color='skyblue', alpha=0.6)
     ax.set_yticklabels([])
     plt.title('Radar Chart')
     plt.show()
     ```

2. **Categorical Data**:
   - Bar Charts:
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt

     # Sample data
     data = {'Category': ['A', 'B', 'C', 'D'],
             'Count': [10, 20, 15, 25]}
     df = pd.DataFrame(data)

     # Plot bar chart
     plt.bar(df['Category'], df['Count'], color='skyblue')
     plt.title('Bar Chart')
     plt.xlabel('Category')
     plt.ylabel('Count')
     plt.show()
     ```

   - Pie Chart:
     ```python
     import matplotlib.pyplot as plt

     # Sample data
     sizes = [15, 30, 45, 10]
     labels = ['A', 'B', 'C', 'D']

     # Plot pie chart
     plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
     plt.title('Pie Chart')
     plt.show()
     ```

   - Stacked Bar Chart:
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt

     # Sample data
     data = {'Category': ['A', 'B', 'C'],
             'Group1': [10, 20, 15],
             'Group2': [15, 25, 20]}
     df = pd.DataFrame(data)

     # Plot stacked bar chart
     df.plot(x='Category', kind='bar', stacked=True)
     plt.title('Stacked Bar Chart')
     plt.xlabel('Category')
     plt.ylabel('Count')
     plt.show()
     ```

   - Grouped Bar Chart:
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt

     # Sample data
     data = {'Category': ['A', 'B', 'C', 'D'],
             'Group1': [10, 15, 20, 25],
             'Group2': [15, 20, 25, 30]}
     df = pd.DataFrame(data)

     # Plot grouped bar chart
     df.plot(x='Category', kind='bar')
     plt.title('Grouped Bar Chart')
     plt.xlabel('Category')
     plt.ylabel('Count')
     plt.show()
     ```

3. **Numerical Data**:
   - Histograms:
     ```python
     import numpy as np
     import matplotlib.pyplot as plt

     # Generate random data
     data = np.random.randn(1000)

     # Plot histogram
     plt.hist(data, bins=30, edgecolor='black')
     plt.title('Histogram')
     plt.xlabel('Value')
     plt.ylabel('Frequency')
     plt.show()
     ```

   - Box Plot:
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt

     # Sample data
     data = {'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
             'Value': [10, 15, 20, 25, 30, 35]}
     df = pd.DataFrame(data)

     # Plot box plot
     df.boxplot(by='Category', column='Value')
     plt.title('Box Plot')
     plt.xlabel('Category')
     plt.ylabel('Value')
     plt.show()
     ```

   - Scatter Plot:
     ```python
     import numpy as np
     import matplotlib.pyplot as plt

     # Generate random data
     x = np.random.randn(100)
     y = np.random.randn(100)

     # Plot scatter plot
     plt.scatter(x, y)
     plt.title('Scatter Plot')
     plt.xlabel('X-axis')
     plt.ylabel('Y-axis')
     plt.show()
     ```

   - Line Plot:
     ```python
     import numpy as np
     import matplotlib.pyplot as plt

     # Generate random data
     x = np.linspace(0, 10, 100)
     y = np.sin(x)

     # Plot line plot
     plt.plot(x, y)
     plt.title('Line Plot')
     plt.xlabel('X-axis')
     plt.ylabel('Y-axis')
     plt.show()
     ```

   - Heatmap:
     ```python
     import numpy as np
     import seaborn as sns
     import matplotlib.pyplot as plt

     # Generate random data
     data = np
