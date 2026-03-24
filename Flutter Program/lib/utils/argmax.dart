//Function to get mask

int argmax(List<double> a) {
  var bestI = 0;
  var bestV = -double.infinity;
  for (var i = 0; i < a.length; i++) {
    if (a[i] > bestV) {
      bestV = a[i];
      bestI = i;
    }
  }
  return bestI;
}
