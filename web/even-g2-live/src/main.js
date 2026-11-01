import { waitForEvenAppBridge, CreateStartUpPageContainer, TextContainerProperty, TextContainerUpgrade, DeviceConnectType, OsEventTypeList } from '@evenrealities/even_hub_sdk';
import { CaptureSession, launchConfiguration, waitForNativeBridge } from './capture-session.js';
import './style.css';

const $ = id => document.getElementById(id);
let session = null;
let latest = null;
let renderedState = '';
let renderTimer = null;
const stateLabels = { idle: 'Ready for G2', connecting: 'Connecting companion', starting_mic: 'Opening G2 microphone', recording: 'Capturing G2 microphone', stopping: 'Stopping microphone', draining: 'Finishing enhancement', stopped: 'Capture saved', error: 'Capture stopped' };
const messages = {
  idle: 'Press Start when you are ready. No microphone is open.',
  connecting: 'Checking the connected G2 and preparing local processing.',
  starting_mic: 'Waiting for the official Even app to confirm the glasses microphone.',
  recording: 'Speak near the glasses. Audio goes only to the companion on this phone.',
  stopping: 'Closing the G2 microphone and waiting for its final audio callback.',
  draining: 'Saving the remaining enhanced audio and filter tail.',
  stopped: 'Open the Android companion to review the raw and enhanced recordings.',
};

function render(snapshot) {
  $('state').textContent = stateLabels[snapshot.state] ?? snapshot.state;
  $('message').textContent = snapshot.error || messages[snapshot.state] || '';
  if (snapshot.state === 'stopped' && !snapshot.liveInputVerified) {
    $('state').textContent = 'Stopped — no G2 audio';
    $('message').textContent = 'No live G2 PCM was verified in this session. Check the glasses connection and reopen this plugin.';
  }
  $('lamp').className = `lamp ${snapshot.state}`;
  $('start').disabled = !['idle', 'stopped', 'error'].includes(snapshot.state);
  $('stop').disabled = !['connecting', 'starting_mic', 'recording'].includes(snapshot.state);
  $('input').textContent = `${(snapshot.sentPayloadBytes / 2).toLocaleString()} samples`;
  $('output').textContent = `${(snapshot.enhancedPayloadBytes / 2).toLocaleString()} samples`;
  $('events').textContent = snapshot.callbacks.toLocaleString();
  $('continuity').textContent = snapshot.discontinuities ? 'Interrupted — see details' : snapshot.callbacks ? 'No client discontinuity' : 'Waiting for audio';
  $('details').textContent = JSON.stringify(snapshot, null, 2);
  renderedState = snapshot.state;
}

function changed(snapshot) {
  latest = snapshot;
  if (snapshot.state !== renderedState) { clearTimeout(renderTimer); renderTimer = null; render(snapshot); }
  else if (!renderTimer) renderTimer = setTimeout(() => { renderTimer = null; render(latest); }, 250);
}

async function boot() {
  let config;
  try { config = launchConfiguration(location.href); }
  catch (error) { $('message').textContent = error.message; return; }
  const waitingTimer = setTimeout(() => {
    $('message').textContent = 'Still waiting for the Even app bridge. A normal browser cannot capture the G2 microphone.';
  }, 5000);
  await waitForNativeBridge(window);
  const bridge = await waitForEvenAppBridge();
  clearTimeout(waitingTimer);
  const result = await bridge.createStartUpPageContainer(new CreateStartUpPageContainer({
    containerTotalNum: 1,
    textObject: [new TextContainerProperty({
      containerID: 1, containerName: 'reuse-live', xPosition: 0, yPosition: 0,
      width: 576, height: 288, paddingLength: 8, borderWidth: 0, isEventCapture: 1,
      content: 'RE-USE G2 microphone\n\nStart capture on your phone.\nDouble tap to stop.',
    })],
  }));
  if (result !== 0) throw new Error(`Glasses startup page failed (${result}). Reopen this plugin before trying again.`);
  session = new CaptureSession({ bridge, socketUrl: config.socketUrl, connectedValue: DeviceConnectType.Connected,
    visible: () => !document.hidden, onChange: changed });
  changed(session.snapshot());
  $('start').addEventListener('click', () => { void session.start().catch(() => {}); });
  $('stop').addEventListener('click', () => { void session.stop('user').catch(() => {}); });
  document.addEventListener('visibilitychange', () => { void session.visibilityChanged(document.hidden).catch(() => {}); });
  window.addEventListener('pagehide', () => { void session.stop('pagehide').catch(() => {}); });
  bridge.onEvenHubEvent(event => {
    if (event.textEvent?.eventType === OsEventTypeList.DOUBLE_CLICK_EVENT) {
      void session.stop('glasses-double-tap').catch(() => {});
    }
  });
  // Counters only: there is no production hook for synthetic PCM or phone audio.
  window.__reuseLiveDiagnostics = () => session.snapshot();
  // Keep glass updates out of the audio callback path; the phone shows counters.
  void bridge.textContainerUpgrade(new TextContainerUpgrade({ containerID: 1, containerName: 'reuse-live',
    content: 'RE-USE ready\n\nUse Start / Stop on your phone.\nGlasses microphone only.' })).catch(() => {});
}

void boot().catch(error => {
  $('state').textContent = 'Unable to initialize'; $('lamp').className = 'lamp error'; $('message').textContent = error.message;
  $('start').disabled = true; $('stop').disabled = true;
});
