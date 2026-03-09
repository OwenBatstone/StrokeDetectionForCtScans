import 'package:flutter/material.dart';
import '../data_classes/paitient_summary.dart';

import '../stroke_inference_service/stroke_inference_service.dart';

class PatientSummaryCard extends StatelessWidget { //for showing the final prediction
  final PatientSummary summary;
  const PatientSummaryCard({required this.summary}); //summary data about patient

  @override
  Widget build(BuildContext context) {
    final probs = summary.perClassProb;
    final labels = StrokeInferenceService.labels;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Overall prediction',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            Text(
              '${summary.label} • ${(summary.confidence * 100).toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 6),
            Text('Slices used: ${summary.slicesUsed}/${summary.totalSlices}'),
            const SizedBox(height: 10),
            for (int i = 0; i < labels.length; i++)
              Text('${labels[i]}: ${(probs[i] * 100).toStringAsFixed(1)}%'),
          ],
        ),
      ),
    );
  }
}