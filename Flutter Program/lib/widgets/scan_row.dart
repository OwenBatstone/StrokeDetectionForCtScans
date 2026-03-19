import 'package:flutter/material.dart';
import '../data_classes/paitient_summary.dart';   // ScanRecord
import '../widgets/slice_card.dart';           // SliceCard


class ScanRow extends StatelessWidget { 
  const ScanRow({required this.scan, super.key}); 

  final PatientSummary scan; 

  @override
  Widget build(BuildContext context){ 
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12), 
        child: Column (
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.folder_open, size: 18 ), 
              const SizedBox(width: 8),
              Text('Scan : ${scan.scanId}',
              style: Theme.of(context)
              .textTheme
              .titleMedium
              ?.copyWith(fontWeight: FontWeight.bold),
              ),
              const Spacer(), 
              Text(
                '${scan.slicesIds.length} slices', 
                style: Theme.of(context).textTheme.bodySmall, 
              )

            ],
          ), 

          const SizedBox(height: 12,),
          const Divider(height: 1,), 
          const SizedBox(height: 12), 

          InfoRow(label: 'Prediction', value: scan.label,),
          InfoRow(label: 'Confidence', value: scan.confidence.toString()),
          InfoRow(label: 'Total Slices', value: scan.totalSlices.toString()),
          InfoRow(label: 'Slices Used', value: scan.slicesUsed.toString()),
          InfoRow(label: 'Run By', value: scan.run_by ?? 'empty'),

          SizedBox(
            height: 200,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: scan.imageUrl.length,
              separatorBuilder: (_, __) => const SizedBox(width: 10,),
              itemBuilder: (context , i) =>SliceCard(
                sliceId : scan.slicesIds[i], 
                imageUrl : scan.imageUrl[i]
                ),
            ),
          )
        ],
        ),
        ),
      );
  }
}


class InfoRow extends StatelessWidget { 
  const InfoRow({required this.label, required this.value, super.key}); 

  final String label; 
  final String value;

  @override
  Widget build(BuildContext context) { 
    return Padding (
      padding: const EdgeInsets.symmetric(vertical : 3),
      child: Row( 
        children: [
          Text(
            '$label : ', 
            style : Theme.of(context)
            .textTheme
            .bodySmall
            ?.copyWith(fontWeight: FontWeight.w600),
          ),
          Expanded(child: Text(
            value, 
            style: Theme.of(context).textTheme.bodySmall,
            overflow: TextOverflow.ellipsis,
          ))
        ],
      ),
    );
  }

}