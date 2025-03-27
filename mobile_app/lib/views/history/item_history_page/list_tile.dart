import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class MyListTile extends StatelessWidget {
  final String title;
  final String imageLink;
  final String time;
  MyListTile({super.key, required this.title, required this.imageLink, required this.time});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: Image.network(imageLink),
        title: Text(title),
        subtitle: Text(time),
      ),
    );
  }
}
