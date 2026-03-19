import 'package:flutter/material.dart';
import '../data_classes/paitient_summary.dart';          // ScanRecord
import '../supabase_functions/get_scan_records.dart'; // GetScanRecords
import '../widgets/scan_row.dart';                    // ScanRow


class AdminView extends StatefulWidget {
  const AdminView({super.key});

  @override
  State<AdminView> createState() => AdminViewState();
}

class AdminViewState extends State<AdminView> {
  final GetScanRecords _retrival = GetScanRecords(); 
  late Future<List<PatientSummary>> _future; 

  @override
  void initState(){ 
    super.initState();
    _future = _retrival.fetchAllScansWithSlices();

  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Stroke Zip Classifier + Locator'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => setState(() {
              _future = _retrival.fetchAllScansWithSlices();
            }),
            ),
        ],
      ),
      body: FutureBuilder<List<PatientSummary>>(
        future: _future, 
        builder: (context, snapshot){
          if(snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text('ERROR : ${snapshot.error}'));
          }

          final scans = snapshot.data ?? []; 

          if (scans.isEmpty) {
            return const Center(child: Text('No Scans Found')) ; 
          
          }

          return SizedBox(
            height: 800, 
            child: ListView.separated(
              scrollDirection: Axis.vertical,
              padding: const EdgeInsets.all(16),
              itemCount: scans.length,
              separatorBuilder: (_, __) => const SizedBox(width : 320),
                itemBuilder: (context, index) => SizedBox(
                  width: 200,
                  child: ScanRow(scan: scans[index])
                ), 
              
              ),
              
            );
            
          



        },
      ),
    );
  
  }
}