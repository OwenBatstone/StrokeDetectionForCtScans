//File For Admin Page.
//contatins page layout
//individual components located in widget folder

import 'package:flutter/material.dart';
import 'package:stroketry3/pages/stroke_zip_home.dart';
import '../data_classes/paitient_summary.dart';
import '../supabase_functions/get_scan_records.dart';
import '../widgets/scan_row.dart';
import '../supabase_functions/is_user_admin.dart';

class AdminView extends StatefulWidget {
  const AdminView({super.key});

  @override
  State<AdminView> createState() => AdminViewState();
}

class AdminViewState extends State<AdminView> {
  final GetScanRecords _retrival =
      GetScanRecords(); //variable for scan data retrival
  late Future<List<PatientSummary>>
  _future; // type List of list for paitient summary
  bool _isAdmin = false;

  @override
  void initState() {
    super.initState();
    _future = _retrival.fetchAllScansWithSlices(); // future fetch scans
    _loadAdminStatus();
  }

  //function to retrive privilages of user
  Future<void> _loadAdminStatus() async {
    try {
      final adminStatus =
          await isAdmin(); //call isAdmin ../supabase_functions/is_user_admin
      if (mounted) {
        // if widget is mounted set state varible for admin access
        setState(() {
          _isAdmin = adminStatus;
        });
      }
    } catch (e) {
      print('isAdmin error: $e');
      if (mounted)
        setState(() => _isAdmin = false); // if admin is false set variable
    }
  }

  //build for page
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        //const top bar
        title: const Text('Stroke Zip Classifier + Locator'),
        actions: [
          // buttons top of Page
          IconButton(
            icon: const Icon(
              Icons.refresh,
            ), //refresh button to allow for refreshing database query
            onPressed: () => setState(() {
              _future = _retrival.fetchAllScansWithSlices();
            }),
          ),
          //Button for Admin View Toggle
          if (_isAdmin) ...[
            // if user has admin privliges have button to navigate to and from adminview
            FilledButton(
              onPressed: () => Navigator.push(
                //push home context to navigate back home
                context,
                MaterialPageRoute(builder: (context) => const StrokeZipHome()),
              ),
              child: const Text("Home Page"), // Button Text
            ),
            const SizedBox(width: 10), //size of text box for button
          ],
        ],
      ),
      //Admin view Cards built here
      body: FutureBuilder<List<PatientSummary>>(
        future: _future, // data retrival
        builder: (context, snapshot) {
          //snapshot to allow for handling async
          if (snapshot.connectionState == ConnectionState.waiting) {
            // if waiting for data show loading icon
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            // for error show error
            return Center(child: Text('ERROR : ${snapshot.error}'));
          }

          final scans = snapshot.data ?? []; // scans data

          if (scans.isEmpty) {
            //for case empty scan
            return const Center(child: Text('No Scans Found'));
          }

          return ListView.separated(
            //List view for carts
            padding: const EdgeInsets.all(16), //Gap Between
            itemCount:
                scans.length, //number of cards = length of return from query
            separatorBuilder: (_, __) =>
                const SizedBox(height: 12), //build sized bxes
            itemBuilder: (context, index) => ScanRow(
              scan: scans[index],
            ), // item builder using widget  scan row in ../widget/scanrow
          );
        },
      ),
    );
  }
}
