# Conditional read procedure

If lookup_service returns ready=false, finish the read-only inspection without requesting device health or alarms.
Otherwise call get_device_health with the device ID returned by lookup_service.
If health.available=true, finish the read-only inspection without retrieving an alarm.
Otherwise use health.alarmId from that health response to call get_alarm.
After retrieving the alarm, hand off to L1 to explain the service health and selected alarm code. Apply the output restrictions in the parent Skill; suggest possible investigation but never execute changes.
